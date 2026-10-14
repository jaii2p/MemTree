"""Distill Phi-4 Mini's original attention into the converted layers.

Stage "layerwise": every converted layer sees the exact input the original
layer saw (teacher forcing) and is trained to reproduce its output (relative
MSE). Cheap, stable, and all layers train in parallel from one teacher pass.

Stage "kl": the whole converted model is trained to match the original
model's next-token distribution (KL), so errors that compound across layers
get corrected.

Only the new parameters train (tree mixer weights, feature maps, gates);
Phi's own weights stay frozen and are shared with the teacher.

    python -m memtree.distill --data corpus.txt --stage both --steps 300 \
        --seq 1024 --window 256 --out adapter.pt

Use sequences clearly longer than --window: below the window the converted
layer already equals the original, so there is nothing to learn there.
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .phi_adapter import (convert_phi3, converted_layers, load_adapter,
                          new_parameters, save_adapter, teacher_mode)


def tokenize_file(tokenizer, path, block_chars=200_000, batch=32):
    """Tokenize a large text file in paragraph-aligned blocks (fast batched
    encoding, bounded memory) and return one flat list of ids."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    groups, cur, size = [], [], 0
    for para in text.split("\n\n"):
        cur.append(para)
        size += len(para) + 2
        if size >= block_chars:
            groups.append("\n\n".join(cur) + "\n\n")
            cur, size = [], 0
    if cur:
        groups.append("\n\n".join(cur))
    parts = []
    for i in range(0, len(groups), batch):
        for row in tokenizer(groups[i:i + batch], add_special_tokens=False)["input_ids"]:
            parts.append(np.asarray(row, dtype=np.int32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def token_chunks(tokenizer, path, seq_len, seed=0):
    """Returns (held_out, endless training stream). The held-out chunk is never
    trained on (unless the corpus is a single chunk, which is reported)."""
    ids = tokenize_file(tokenizer, path)
    print(f"corpus: {len(ids):,} tokens -> {len(ids) // seq_len:,} sequences of {seq_len}")
    n = len(ids) // seq_len
    if n == 0:
        raise ValueError(f"corpus has {len(ids)} tokens, fewer than one sequence of {seq_len}")
    chunks = ids[:n * seq_len].reshape(n, seq_len)  # a view, no copy
    held = torch.from_numpy(chunks[-1:].astype(np.int64))
    order = list(range(n - 1)) or [0]
    if n == 1:
        print("warning: corpus is one sequence long; held-out eval overlaps training")

    def stream():
        rng = random.Random(seed)
        while True:
            rng.shuffle(order)
            for i in order:
                yield torch.from_numpy(chunks[i:i + 1].astype(np.int64))
    return held, stream()


def freeze_all_but_adapter(model):
    for p in model.parameters():
        p.requires_grad_(False)
    params = new_parameters(model)
    for p in params:
        p.requires_grad_(True)
    return params


def capture_teacher_io(model, input_ids):
    """One original-attention forward; record each converted layer's input,
    rotary embeddings and output."""
    records, hooks = {}, []
    for m in converted_layers(model):
        def hook(mod, args, kwargs, output, _m=m):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            pe = tuple(t.detach() for t in kwargs["position_embeddings"])
            records[id(_m)] = (hs.detach(), pe, output[0].detach())
        hooks.append(m.register_forward_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad(), teacher_mode(model):
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return records


def layerwise_step(model, input_ids, opt, params):
    rec = capture_teacher_io(model, input_ids)
    opt.zero_grad(set_to_none=True)
    losses = []
    for m in converted_layers(model):
        hs, pe, target = rec[id(m)]
        out, _ = m(hs, position_embeddings=pe)
        t = target.float()
        loss = F.mse_loss(out.float(), t) / (t.pow(2).mean() + 1e-8)
        loss.backward()  # per layer: graphs are independent, frees memory early
        losses.append(loss.item())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return sum(losses) / len(losses)


def kl_loss(student_logits, teacher_logits, tau=1.0, chunk=256):
    T = student_logits.shape[1]
    total = 0.0
    for i in range(0, T, chunk):
        ls = F.log_softmax(student_logits[:, i:i + chunk].float() / tau, dim=-1)
        lt = F.log_softmax(teacher_logits[:, i:i + chunk].float() / tau, dim=-1)
        total = total + F.kl_div(ls, lt, log_target=True, reduction="sum")
    return total * tau * tau / (student_logits.shape[0] * T)


def kl_step(model, input_ids, opt, params, tau=1.0):
    with torch.no_grad(), teacher_mode(model):
        t_logits = model(input_ids=input_ids, use_cache=False).logits
    opt.zero_grad(set_to_none=True)
    s_logits = model(input_ids=input_ids, use_cache=False).logits
    loss = kl_loss(s_logits, t_logits, tau)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, input_ids):
    """KL(teacher || student) per token and how open the tree gates are."""
    with teacher_mode(model):
        t = model(input_ids=input_ids, use_cache=False).logits
    s = model(input_ids=input_ids, use_cache=False).logits
    gates = torch.cat([torch.tanh(m.gate.detach().float()).abs() for m in converted_layers(model)])
    agree = (t.argmax(-1) == s.argmax(-1)).float().mean().item()
    return {"kl": kl_loss(s, t).item(), "top1_agree": agree,
            "gate_mean": gates.mean().item(), "gate_max": gates.max().item()}


def load_model(name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval(), tok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--data", required=True, help="plain-text training corpus")
    ap.add_argument("--stage", choices=["layerwise", "kl", "both"], default="both")
    ap.add_argument("--steps", type=int, default=300, help="steps per stage")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-global", type=int, default=4)
    ap.add_argument("--max-levels", type=int, default=17)
    ap.add_argument("--keep-full", type=int, nargs="*", default=[])
    ap.add_argument("--lr-layerwise", type=float, default=1e-3)
    ap.add_argument("--lr-kl", type=float, default=1e-4)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--out", default="adapter.pt")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing for the KL stage (less memory, ~30%% slower)")
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    model, tok = load_model(a.model, a.device, getattr(torch, a.dtype))
    if a.init_adapter:
        load_adapter(model, a.init_adapter)
    else:
        convert_phi3(model, a.window, a.n_global, a.max_levels, a.keep_full)
    params = freeze_all_but_adapter(model)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF only checkpoints in train mode; Phi has no dropout, so outputs match
    print(f"converted {len(list(converted_layers(model)))} layers, "
          f"{sum(p.numel() for p in params):,} trainable parameters")

    held_out, data = token_chunks(tok, a.data, a.seq)
    held_out = held_out.to(a.device)
    print("init", evaluate(model, held_out))
    stages = ["layerwise", "kl"] if a.stage == "both" else [a.stage]
    for stage in stages:
        lr = a.lr_layerwise if stage == "layerwise" else a.lr_kl
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        step_fn = layerwise_step if stage == "layerwise" else kl_step
        t0 = time.time()
        for step in range(1, a.steps + 1):
            loss = step_fn(model, next(data).to(a.device), opt, params)
            if step % 10 == 0:
                print(f"[{stage}] step {step} loss {loss:.5f} ({time.time() - t0:.0f}s)")
            if step % a.eval_every == 0 or step == a.steps:
                print(f"[{stage}] eval", evaluate(model, held_out))
                save_adapter(model, a.out)
    print(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
