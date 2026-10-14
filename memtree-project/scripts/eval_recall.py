"""Long-context recall test: does the tree carry what the window drops?

A fact ("The secret code for Marisol is 48213.") is planted at some depth in
held-out book text, and the model is asked for it at the end. Every planted
fact is scored on three versions of the same model:

  original      Phi's own full attention (quadratic: the ceiling)
  window-only   the converted model with every tree gate shut (the floor)
  tree          the converted model as distilled (window + tree)

If tree beats window-only at lengths past the window, the tree is carrying
long-range memory. The table also shows the attention reads each version
needs, relative to the original.

    python scripts/eval_recall.py --adapter adapter.pt --haystack haystack.txt \
        --lengths 512 1024 2048 4096 --depths 0.1 0.5 0.9 --trials 10
"""
import argparse
import contextlib
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAMES = ["Marisol", "Teodor", "Anouk", "Bartholomew", "Keiko", "Oluwaseun", "Greta",
         "Ishaan", "Lucinda", "Rafferty", "Svetlana", "Desmond", "Priya", "Ezekiel"]


# ------------------------------------------------------------------ pure parts
def insert_needle(text: str, needle: str, depth: float) -> str:
    """Put the needle at a sentence boundary near `depth` (0 = start, 1 = end)."""
    pos = int(len(text) * min(max(depth, 0.0), 1.0))
    m = re.compile(r"[.!?][\"')\]]?\s").search(text, pos)
    cut = m.end() if m else len(text)
    return text[:cut] + needle + " " + text[cut:]


def make_trial(rng: random.Random, filler_text: str, depth: float):
    name = rng.choice(NAMES)
    code = str(rng.randint(10000, 99999))
    needle = f"The secret code for {name} is {code}."
    doc = insert_needle(filler_text, needle, depth)
    question = f"What is the secret code for {name}? Reply with just the number."
    return {"doc": doc, "question": question, "answer": code, "needle": needle}


def is_correct(output: str, answer: str) -> bool:
    return answer in re.sub(r"[\s,]", "", output)


def attention_reads(T: int, window: int, n_global: int):
    """Keys/states read per layer for a T-token prompt, by version."""
    full = T * (T + 1) // 2
    sparse = sum(min(t + 1, window) + min(n_global, max(0, t - window + 1)) for t in range(T))
    tree = sparse + sum(bin(t).count("1") + 1 for t in range(T))
    return {"original": full, "window-only": sparse, "tree": tree}


# ------------------------------------------------------------------ model parts
@contextlib.contextmanager
def gates_shut(model):
    import torch
    from memtree.phi_adapter import converted_layers
    mods = list(converted_layers(model))
    saved = [m.gate.detach().clone() for m in mods]
    with torch.no_grad():
        for m in mods:
            m.gate.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for m, g in zip(mods, saved):
                m.gate.copy_(g)


def build_prompt(tok, trial, device):
    msgs = [{"role": "user", "content": trial["doc"] + "\n\n" + trial["question"]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = ids if hasattr(ids, "shape") else ids["input_ids"]
    return ids.to(device)


def answer(model, tok, ids, max_new_tokens=10):
    import torch
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--adapter", required=True, help="adapter.pt from memtree.distill")
    ap.add_argument("--haystack", default="haystack.txt", help="held-out text (make_corpus.py)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=10, help="per length x depth cell")
    ap.add_argument("--versions", nargs="+", default=["original", "window-only", "tree"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from memtree.distill import load_model, tokenize_file
    from memtree.phi_adapter import load_adapter, teacher_mode
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, device, getattr(torch, a.dtype))
    load_adapter(model, a.adapter)
    cfg = model.config.memtree
    print(f"adapter: window {cfg['window']}, {cfg['n_global']} global tokens")

    hay = tokenize_file(tok, a.haystack)
    rng = random.Random(a.seed)
    overhead = 40  # chat template + question, in tokens
    cells = {}
    for L in a.lengths:
        for depth in a.depths:
            trials = []
            for _ in range(a.trials):
                n = L - overhead
                if len(hay) <= n:
                    sys.exit(f"haystack has {len(hay)} tokens; need more than {n}")
                start = rng.randrange(0, len(hay) - n)
                filler = tok.decode(hay[start:start + n].tolist(), skip_special_tokens=True)
                trials.append(make_trial(rng, filler, depth))
            cells[(L, depth)] = trials

    modes = {"original": lambda: teacher_mode(model), "window-only": lambda: gates_shut(model),
             "tree": contextlib.nullcontext}
    results = {}
    t0 = time.time()
    for (L, depth), trials in cells.items():
        for v in a.versions:
            right = 0
            with modes[v]():
                for tr in trials:
                    right += is_correct(answer(model, tok, build_prompt(tok, tr, device)), tr["answer"])
            results[(L, depth, v)] = right / len(trials)
        row = "  ".join(f"{v} {results[(L, depth, v)]:.0%}" for v in a.versions)
        print(f"L={L:>5} depth={depth:.1f}: {row}   ({time.time() - t0:.0f}s)", flush=True)

    print("\nRecall by length (averaged over depths)")
    head = f"{'length':>7} " + " ".join(f"{v:>12}" for v in a.versions)
    print(head + f" {'reads vs original (window-only / tree)':>40}")
    for L in a.lengths:
        accs = [np.mean([results[(L, d, v)] for d in a.depths]) for v in a.versions]
        r = attention_reads(L, cfg["window"], cfg["n_global"])
        rel = f"{r['window-only'] / r['original']:.1%} / {r['tree'] / r['original']:.1%}"
        print(f"{L:>7} " + " ".join(f"{x:>12.0%}" for x in accs) + f" {rel:>40}")
    w = cfg["window"]
    print(f"\nThe planted fact sits outside the {w}-token window when it is more than "
          f"{w} tokens from the end; those are the cells where tree vs window-only matters.")


if __name__ == "__main__":
    main()
