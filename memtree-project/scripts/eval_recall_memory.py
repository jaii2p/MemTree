"""Recall test for the real architecture: linear-context Phi + the watchdog.

The same planted-fact task as eval_recall.py, but each long passage is pasted
as a chat message, so overflow filing sends what will not fit the window out
to the watchdog. Two versions answer every trial:

  original   plain Phi, the whole passage in full quadratic attention
  memtree    Phi with her tree attention (or window-only), fine-tuned for the
             protocol, with the memory tree: the whole message stays in her
             view, overflow is copied onto the tree, whispers, references

    python scripts/eval_recall_memory.py --adapter adapter_v3.pt --finetune finetune_v4.pt \\
        --lengths 2048 4096 8192 --depths 0.1 0.5 0.9 --trials 5
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_recall import is_correct, make_trial  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="richardyoung/Phi-4-mini-instruct-heretic")
    ap.add_argument("--finetune", required=True, help="finetune .pt trained for the window model")
    ap.add_argument("--adapter", default=None, help="tree adapter (adapter_v3.pt); omit for window-only")
    ap.add_argument("--window", type=int, default=1024, help="window-only mode, when no --adapter")
    ap.add_argument("--haystack", default="haystack.txt")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from memtree.distill import load_model, tokenize_file
    from memtree.finetune import load_finetune
    from memtree.phi_adapter import load_adapter, window_model
    from memtree.runtime import MemoryRuntime
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(a.model, device, getattr(torch, a.dtype))

    hay = tokenize_file(tok, a.haystack)
    rng = random.Random(a.seed)
    cells = {}
    for L in a.lengths:
        for d in a.depths:
            trials = []
            for _ in range(a.trials):
                n = L - 40
                start = rng.randrange(0, len(hay) - n)
                filler = tok.decode(hay[start:start + n].tolist(), skip_special_tokens=True)
                trials.append(make_trial(rng, filler, d))
            cells[(L, d)] = trials

    res, t0 = {}, time.time()
    if not a.skip_original:
        for (L, d), trials in cells.items():
            right = 0
            for tr in trials:
                msgs = [{"role": "user", "content": tr["doc"] + "\n\n" + tr["question"]}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                ids = (ids if hasattr(ids, "shape") else ids["input_ids"]).to(device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=16, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                right += is_correct(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), tr["answer"])
            res[(L, d, "original")] = right / len(trials)
            print(f"original L={L} depth={d:.1f}: {res[(L, d, 'original')]:.0%} ({time.time() - t0:.0f}s)", flush=True)

    if a.adapter:
        load_adapter(model, a.adapter)  # Phi's tree attention, math unchanged
    else:
        window_model(model, window=a.window)
    load_finetune(model, tok, a.finetune)
    win = model.config.memtree["window"]
    how = {"reference": 0, "whisper_or_visible": 0}
    for (L, d), trials in cells.items():
        right = 0
        for tr in trials:
            rt = MemoryRuntime(model, tok, window=win)  # fresh memory per trial
            r = rt.chat("me", tr["doc"] + "\n\n" + tr["question"], max_new_tokens=48)
            ok = is_correct(r["reply"], tr["answer"])
            right += ok
            if ok:
                how["reference" if r["references"] else "whisper_or_visible"] += 1
        res[(L, d, "memtree")] = right / len(trials)
        print(f"memtree  L={L} depth={d:.1f}: {res[(L, d, 'memtree')]:.0%} ({time.time() - t0:.0f}s)", flush=True)

    versions = ([] if a.skip_original else ["original"]) + ["memtree"]
    print("\nRecall by length (averaged over depths)")
    print(f"{'length':>7} " + " ".join(f"{v:>10}" for v in versions) + f" {'attention reads, memtree vs original':>40}")
    for L in a.lengths:
        accs = [np.mean([res[(L, d, v)] for d in a.depths]) for v in versions]
        rel = (L * min(L, win) + (L * L.bit_length() if a.adapter else 0)) / (L * (L + 1) / 2)
        print(f"{L:>7} " + " ".join(f"{x:>10.0%}" for x in accs) + f" {rel:>39.1%}")
    print(f"\nmemtree correct answers: {how['reference']} via a reference, "
          f"{how['whisper_or_visible']} from whispers or visible text")


if __name__ == "__main__":
    main()
