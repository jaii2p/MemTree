"""Fine-tune Phi to use the memory protocol.

What it learns:
  * when something it needs is not in fresh memory, write <|mem_ref|>key<|/mem_ref|>
  * after the watchdog's <|mem|> ... <|/mem|> delivery, answer from it
  * when the answer is already in fresh memory, answer directly (no reference)

What trains (Phi's own weights stay frozen):
  * LoRA adapters on qkv_proj, o_proj, gate_up_proj, down_proj
  * small trainable deltas for the 4 memory tokens' input and output embeddings
    (cheaper than training the whole 200k-row tied embedding matrix)

Loss is computed only on what Phi itself should write. The prompt and the
watchdog's deliveries are masked out, since the watchdog inserts those at
runtime.

Data, mixed freely:
  --synthetic N     N examples from memtree.bench sessions (teaches the protocol)
  --data a.jsonl    your own examples, one JSON object per line, either
                    {"system": ..., "user": ..., "assistant": ...} or
                    {"messages": [{"role": ..., "content": ...}, ..., assistant last]}
  Mix in ordinary chat data with --data so Phi keeps its general ability.

    python -m memtree.finetune --adapter adapter.pt --synthetic 3000 \
        --data my_chats.jsonl --steps 1500 --out finetune.pt

Load for chatting:  load_finetune(model, tokenizer, "finetune.pt")
Order matters: distilled adapter first (load_adapter), then the fine-tune.
"""
import argparse
import functools
import json
import math
import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .runtime import MEM, MEM_END, MEM_REF, MEM_REF_END, MEM_TOKENS, add_memory_tokens

DEFAULT_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
_DELIVERY = re.compile("(" + re.escape(MEM) + ".*?" + re.escape(MEM_END) + ")", re.S)


# ------------------------------------------------------------- data (pure python)
def segments(assistant: str) -> List[Tuple[str, bool]]:
    """Split a target into (text, trainable). Watchdog deliveries are not
    trainable: Phi never writes them, the watchdog splices them in."""
    out = []
    for part in _DELIVERY.split(assistant):
        if part:
            out.append((part, not (part.startswith(MEM) and part.endswith(MEM_END))))
    return out


def to_messages(ex: Dict) -> Tuple[List[Dict], str]:
    """Normalize an example to (prompt messages, assistant target)."""
    if "messages" in ex:
        msgs = list(ex["messages"])
        if not msgs or msgs[-1]["role"] != "assistant":
            raise ValueError("last message must be the assistant's")
        return msgs[:-1], msgs[-1]["content"]
    msgs = []
    if ex.get("system"):
        msgs.append({"role": "system", "content": ex["system"]})
    msgs.append({"role": "user", "content": ex["user"]})
    return msgs, ex["assistant"]


def chat_prompt_ids(tok, messages) -> List[int]:
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def end_token_id(tok) -> int:
    """Phi-4 ends assistant turns with <|end|>; fall back to eos."""
    tid = tok.convert_tokens_to_ids("<|end|>")
    unk = getattr(tok, "unk_token_id", None)
    return tid if isinstance(tid, int) and tid >= 0 and tid != unk else tok.eos_token_id


def encode_example(tok, ex: Dict, max_len: int) -> Optional[Tuple[List[int], List[int]]]:
    """(input_ids, labels) with -100 on the prompt and on deliveries.
    Returns None if the example does not fit in max_len."""
    msgs, target = to_messages(ex)
    ids = chat_prompt_ids(tok, msgs)
    labels = [-100] * len(ids)
    for text, trainable in segments(target):
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if trainable else [-100] * len(t)
    end = end_token_id(tok)
    ids.append(end)
    labels.append(end)
    if len(ids) > max_len:
        return None
    return ids, labels


def synthetic_examples(n: int, seed: int = 0, events_per_session: int = 600) -> List[Dict]:
    from .bench import make_session
    from .runtime import reference_sft_examples
    out: List[Dict] = []
    s = seed
    while len(out) < n:
        out += reference_sft_examples(make_session(seed=s, n_events=events_per_session))
        s += 1
    random.Random(seed).shuffle(out)
    return out[:n]


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wants_reference(ex: Dict) -> bool:
    return to_messages(ex)[1].lstrip().startswith(MEM_REF)


# ------------------------------------------------------------------- torch parts
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


@functools.lru_cache(maxsize=None)
def _build_classes():
    torch, nn, F = _torch()

    class LoRALinear(nn.Module):
        """y = base(x) + (alpha / r) * B(A(dropout(x))); B starts at zero, so the
        wrapped layer is unchanged until training moves it."""

        def __init__(self, base, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad_(False)
            dt = torch.promote_types(base.weight.dtype, torch.float32)
            dev = base.weight.device
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.r, self.alpha, self.scale = r, alpha, alpha / r
            self.drop = nn.Dropout(dropout)
            self.in_features, self.out_features = base.in_features, base.out_features

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            y = self.base(x)
            d = F.linear(F.linear(self.drop(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
            return y + (d * self.scale).to(y.dtype)

    class MemTokenDelta(nn.Module):
        """Trainable offsets for the memory tokens only. Input side: added to
        the embedding of those ids. Output side: extra logit term for those
        ids. Both start at zero."""

        def __init__(self, model, token_ids: Sequence[int]):
            super().__init__()
            emb = model.get_input_embeddings()
            dt = torch.promote_types(emb.weight.dtype, torch.float32)
            dev = emb.weight.device
            n, d = len(token_ids), emb.embedding_dim
            self.register_buffer("ids", torch.tensor(list(token_ids), device=dev), persistent=False)
            self.d_in = nn.Parameter(torch.zeros(n, d, device=dev, dtype=dt))
            self.d_out = nn.Parameter(torch.zeros(n, d, device=dev, dtype=dt))
            self._hooks = [emb.register_forward_hook(self._emb_hook),
                           model.get_output_embeddings().register_forward_hook(self._head_hook)]

        def _emb_hook(self, mod, args, out):
            match = (args[0].unsqueeze(-1) == self.ids).to(self.d_in.dtype)  # (B,T,n)
            return out + (match @ self.d_in).to(out.dtype)

        def _head_hook(self, mod, args, out):
            add = args[0].to(self.d_out.dtype) @ self.d_out.T  # (B,T,n)
            return out.index_add(-1, self.ids, add.to(out.dtype))

        def remove(self):
            for h in self._hooks:
                h.remove()

    return LoRALinear, MemTokenDelta


def apply_lora(model, targets: Sequence[str] = DEFAULT_TARGETS, r: int = 16,
               alpha: float = 32.0, dropout: float = 0.05) -> List:
    torch, nn, _ = _torch()
    LoRALinear, _ = _build_classes()
    wrapped = []
    for _, parent in list(model.named_modules()):
        for name in targets:
            child = getattr(parent, name, None)
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(getattr(parent, name))
    if not wrapped:
        raise ValueError(f"no Linear layers named {targets} found")
    return wrapped


def lora_parameters(model) -> List:
    return [p for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))]


def init_new_token_rows(model, token_ids: Sequence[int]):
    """Start new memory tokens at the mean real-token embedding (deterministic,
    so a saved fine-tune reloads onto the same starting point)."""
    torch, _, _ = _torch()
    emb = model.get_input_embeddings().weight
    n_real = min(token_ids)
    with torch.no_grad():
        mean = emb[:n_real].float().mean(0).to(emb.dtype)
        for i in token_ids:
            emb[i] = mean
        head = model.get_output_embeddings()
        if head is not None and head.weight.data_ptr() != emb.data_ptr():
            hmean = head.weight[:n_real].float().mean(0).to(head.weight.dtype)
            for i in token_ids:
                head.weight[i] = hmean


def prepare(model, tok, r=16, alpha=32.0, dropout=0.05, targets=DEFAULT_TARGETS):
    """Add memory tokens, freeze Phi, attach LoRA + token deltas. Returns the
    list of trainable parameters."""
    _, MemTokenDelta = _build_classes()
    if hasattr(model, "mem_delta"):
        raise RuntimeError("model is already prepared for fine-tuning (hooks would stack)")
    ids = add_memory_tokens(tok, model)
    token_ids = [ids[t] for t in MEM_TOKENS]
    init_new_token_rows(model, token_ids)
    for p in model.parameters():
        p.requires_grad_(False)
    apply_lora(model, targets, r, alpha, dropout)
    model.mem_delta = MemTokenDelta(model, token_ids)
    params = lora_parameters(model) + [model.mem_delta.d_in, model.mem_delta.d_out]
    for p in params:
        p.requires_grad_(True)
    model.config.memtree_finetune = {"r": r, "alpha": alpha, "dropout": dropout,
                                     "targets": list(targets), "token_ids": token_ids}
    return params


def save_finetune(model, path: str):
    torch, _, _ = _torch()
    keep = (".lora_A", ".lora_B", "mem_delta.d_in", "mem_delta.d_out")
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(keep)}
    torch.save({"config": model.config.memtree_finetune, "state": state}, path)


def load_finetune(model, tok, path: str):
    torch, _, _ = _torch()
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    prepare(model, tok, cfg["r"], cfg["alpha"], 0.0, cfg["targets"])
    got = model.config.memtree_finetune["token_ids"]
    if got != cfg["token_ids"]:
        raise ValueError(f"memory token ids {got} differ from training {cfg['token_ids']}")
    params = dict(model.named_parameters())
    missing = set(params[n] for n in params
                  if n.endswith((".lora_A", ".lora_B", "mem_delta.d_in", "mem_delta.d_out")))
    with torch.no_grad():
        for n, t in ck["state"].items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
            missing.discard(params[n])
    if missing:
        raise ValueError(f"{len(missing)} fine-tune tensors missing from {path}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


# ------------------------------------------------------------------ training
def batch_loss(model, ids, labels):
    torch, _, F = _torch()
    logits = model(input_ids=ids, use_cache=False).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def evaluate(model, tok, encoded, examples, device):
    """Held-out loss, plus how often Phi makes the right first move:
    reference when the answer is outside fresh memory, answer directly otherwise."""
    torch, _, _ = _torch()
    ref_id = tok.convert_tokens_to_ids(MEM_REF)
    losses, right, n = [], 0, 0
    model.eval()
    with torch.no_grad():
        for (ids, labels), ex in zip(encoded, examples):
            x = torch.tensor([ids], device=device)
            y = torch.tensor([labels], device=device)
            losses.append(batch_loss(model, x, y).item())
            first = next(i for i, l in enumerate(labels) if l != -100)
            logits = model(input_ids=x[:, :first], use_cache=False).logits[0, -1]
            right += (logits.argmax().item() == ref_id) == wants_reference(ex)
            n += 1
    return {"loss": sum(losses) / max(len(losses), 1), "ref_decision_acc": right / max(n, 1)}


def train(model, tok, train_set, eval_set, params, steps=1000, lr=2e-4, accum=8,
          warmup=50, device="cuda", log_every=10, eval_every=200, out=None, seed=0):
    torch, _, _ = _torch()
    rng = random.Random(seed)
    enc_eval = [encode_example(tok, ex, 10 ** 9) for ex in eval_set]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    order = list(range(len(train_set)))
    pos, t0, history = len(order), time.time(), []
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            ids, labels = train_set[order[pos]]
            pos += 1
            loss = batch_loss(model, torch.tensor([ids], device=device),
                              torch.tensor([labels], device=device)) / accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        history.append(total)
        if step % log_every == 0:
            print(f"step {step} loss {total:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
        if eval_set and (step % eval_every == 0 or step == steps):
            print("eval", evaluate(model, tok, enc_eval, eval_set, device))
            if out:
                save_finetune(model, out)
    model.eval()
    return history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--adapter", default=None, help="distilled adapter.pt from memtree.distill")
    ap.add_argument("--window", type=int, default=None,
                    help="linear-context Phi with this window and no tree (no adapter needed)")
    ap.add_argument("--synthetic", type=int, default=2000)
    ap.add_argument("--data", nargs="*", default=[], help="JSONL files")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    ap.add_argument("--grad-ckpt", action="store_true", help="trade compute for memory")
    ap.add_argument("--out", default="finetune.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch, _, _ = _torch()
    from .distill import load_model
    from .phi_adapter import load_adapter
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    model, tok = load_model(a.model, device, getattr(torch, a.dtype))
    if a.adapter and a.window:
        raise SystemExit("use --adapter (tree version) or --window (no tree), not both")
    if a.adapter:
        load_adapter(model, a.adapter)
        print(f"loaded distilled adapter {a.adapter}")
    elif a.window:
        from .phi_adapter import window_model
        window_model(model, window=a.window)
        print(f"linear context: window {a.window}, no tree; long-range memory is the watchdog's")
    params = prepare(model, tok, a.rank, a.alpha, a.dropout, a.targets)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")

    examples = synthetic_examples(a.synthetic, a.seed) if a.synthetic else []
    for path in a.data:
        examples += read_jsonl(path)
    if not examples:
        raise SystemExit("no training examples: use --synthetic and/or --data")
    random.Random(a.seed).shuffle(examples)
    n_eval = max(1, int(len(examples) * a.eval_frac))
    eval_set, train_raw = examples[:n_eval], examples[n_eval:]
    train_set, skipped = [], 0
    for ex in train_raw:
        enc = encode_example(tok, ex, a.max_len)
        if enc is None:
            skipped += 1
        else:
            train_set.append(enc)
    print(f"{len(train_set)} train / {len(eval_set)} eval examples "
          f"({sum(map(wants_reference, train_raw))} reference, {skipped} too long, skipped)")
    print("before", evaluate(model, tok, [encode_example(tok, ex, 10 ** 9) for ex in eval_set],
                             eval_set, device))
    train(model, tok, train_set, eval_set, params, a.steps, a.lr, a.accum, device=device,
          out=a.out, seed=a.seed)
    save_finetune(model, a.out)
    print(f"saved fine-tune to {a.out}")


if __name__ == "__main__":
    main()
