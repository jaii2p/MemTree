"""Self-test on the real Phi-4 Mini. Run this first on your machine:

    pip install torch transformers
    python run_tests.py                      # unit tests incl. a tiny random Phi3
    python scripts/selftest_phi.py           # this script, ~8 GB in bf16

Checks, in order:
  1. converted == original for a prompt shorter than the window (gate closed)
  2. teacher mode reproduces the original exactly
  3. how far apart they are on a prompt longer than the window, before training
     (this gap is what distillation closes)
  4. KV-cache decode == full forward with the tree branch switched on
  5. greedy generation still produces sensible text
  6. a few layerwise distillation steps lower the loss
  7. the memory runtime runs end to end (Phi has not learned the reference
     protocol yet, so it will not emit references; that needs fine-tuning)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from memtree.distill import (freeze_all_but_adapter, kl_loss, layerwise_step,  # noqa: E402
                             load_model)
from memtree.phi_adapter import (convert_phi3, converted_layers,  # noqa: E402
                                 teacher_mode)

TEXT = ("The lighthouse keeper kept a logbook of every ship that passed, the weather, "
        "and the colour of the sea at dawn. Over forty years the entries changed from "
        "careful copperplate to a hurried scrawl, but he never missed a day. ")


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'WARN'}] {name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    t0 = time.time()
    model, tok = load_model(a.model, a.device, dtype)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s on {a.device} ({a.dtype})")
    results = []

    short = tok(TEXT, return_tensors="pt").input_ids[:, : a.window // 2].to(a.device)
    reps = (3 * a.window) // short.shape[1] + 2
    long_ids = tok(TEXT * reps * 2, return_tensors="pt").input_ids[:, : 3 * a.window].to(a.device)

    with torch.no_grad():
        base_short = model(short, use_cache=False).logits.float()
        base_long = model(long_ids, use_cache=False).logits.float()
    convert_phi3(model, window=a.window, n_global=4, max_levels=17)
    n = len(list(converted_layers(model)))
    print(f"converted {n} layers (window {a.window}, 4 global tokens)")

    with torch.no_grad():
        conv_short = model(short, use_cache=False).logits.float()
        with teacher_mode(model):
            teach_short = model(short, use_cache=False).logits.float()
        conv_long = model(long_ids, use_cache=False).logits.float()

    tol = 1e-3 if dtype == torch.float32 else 0.15  # bf16: different SDPA kernel
    d = (conv_short - base_short).abs().max().item()
    agree = (conv_short.argmax(-1) == base_short.argmax(-1)).float().mean().item()
    results.append(report("1 short prompt matches original", d < tol or agree > 0.99,
                          f"max|dlogit| {d:.4f}, top-1 agreement {agree:.4f}"))
    d = (teach_short - base_short).abs().max().item()
    results.append(report("2 teacher mode is the original", d < 1e-4, f"max|dlogit| {d:.2e}"))
    kl = kl_loss(conv_long, base_long).item()
    agree = (conv_long.argmax(-1) == base_long.argmax(-1)).float().mean().item()
    report("3 long prompt gap before training", True,
           f"KL {kl:.4f}/token, top-1 agreement {agree:.3f} (distillation target)")

    with torch.no_grad():
        for m in converted_layers(model):
            m.gate.fill_(0.2)
        full = model(long_ids, use_cache=False).logits[:, -1].float()
        out = model(long_ids[:, :-1], use_cache=True)
        step = model(long_ids[:, -1:], past_key_values=out.past_key_values, use_cache=True)
        dd = (step.logits[:, -1].float() - full).abs().max().item()
        same = step.logits[:, -1].argmax().item() == full.argmax().item()
        for m in converted_layers(model):
            m.gate.fill_(0.0)
    results.append(report("4 cached decode == full forward", dd < tol or same,
                          f"max|dlogit| {dd:.4f}, same argmax {same}"))

    msgs = [{"role": "user", "content": "In one sentence, what does a lighthouse keeper do?"}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    p = (p if hasattr(p, "shape") else p["input_ids"]).to(a.device)
    with torch.no_grad():
        g = model.generate(p, max_new_tokens=40, do_sample=False)
    text = tok.decode(g[0, p.shape[1]:], skip_special_tokens=True).strip()
    results.append(report("5 generation", len(text) > 10, repr(text[:160])))

    params = freeze_all_but_adapter(model)
    opt = torch.optim.AdamW(params, lr=1e-3)
    losses = [layerwise_step(model, long_ids, opt, params) for _ in range(8)]
    results.append(report("6 layerwise distill lowers loss", losses[-1] < losses[0],
                          " -> ".join(f"{x:.4f}" for x in losses)))
    for p_ in params:
        p_.requires_grad_(False)

    from memtree.runtime import MemoryRuntime
    rt = MemoryRuntime(model, tok, deliver_chars=200)
    r1 = rt.chat("me", "My dog's name is Biscuit and she is scared of thunder.", max_new_tokens=40)
    r2 = rt.chat("me", "What's my dog called?", max_new_tokens=30)
    results.append(report("7 memory runtime", bool(r2["reply"]),
                          f"memory nodes {len(rt.store.branch('me'))}, reply {r2['reply'][:120]!r}"))

    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
