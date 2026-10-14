# memtree

Logarithmic geometric-tree memory, linear sparse context, and a watchdog that
owns every memory operation, built onto Phi-4 Mini.

## Run order on your machine

Renting a GPU? Follow `RUN_ON_VAST.md` step by step.

    pip install torch transformers numpy
    python run_tests.py                 # all unit tests (torch ones included now)
    python scripts/selftest_phi.py      # real Phi-4 Mini: 7 checks
    python -m memtree.distill --data corpus.txt --stage both --steps 300 \
        --seq 1024 --window 256 --out adapter.pt
    python -m memtree.finetune --adapter adapter.pt --synthetic 3000 \
        --data my_chats.jsonl --steps 1500 --out finetune.pt
    PYTHONPATH=. python -m memtree.bench

`corpus.txt` is any plain-text file of a few MB or more. Keep `--seq` well
above `--window`, because below the window the converted layer already
equals the original.

The fine-tuner teaches Phi the reference protocol. `--synthetic` builds
examples from simulated sessions. `--data` takes your own JSONL, either
`{"system", "user", "assistant"}` or `{"messages": [...]}` with the
assistant last. Mix in ordinary chat data so Phi keeps its general ability.
Loss is only on what Phi writes: the prompt and the watchdog's deliveries
are masked. Add `--grad-ckpt` if you run short on GPU memory.

To chat afterwards:

    from memtree.distill import load_model
    from memtree.phi_adapter import load_adapter
    from memtree.finetune import load_finetune
    from memtree.runtime import MemoryRuntime
    model, tok = load_model("microsoft/Phi-4-mini-instruct", "cuda", torch.bfloat16)
    load_adapter(model, "adapter.pt")          # distilled attention first
    load_finetune(model, tok, "finetune.pt")   # then the protocol fine-tune
    rt = MemoryRuntime(model, tok)
    print(rt.chat("me", "My dog is called Biscuit.")["reply"])

## Files

| File | What it is |
|---|---|
| geometry.py | depth = floor(log2(rank+1)); Fenwick partition |
| ref_numpy.py | reference math everything is tested against |
| sparse_attn.py | linear sparse attention (window + globals) |
| tree_mixer.py | log-linear tree mixer, O(T log T), O(log T) state |
| store.py | per-peer branches, log gate (dedup + routing), capped bridges |
| watchdog.py | sole referencer: keeps the center in Phi's context, confirms, stubs, delivers, promotes, sweeps, walks bridges |
| phi_adapter.py | Phi3Attention -> sparse + gated tree attention, on Phi's own weights |
| distill.py | layerwise MSE, then end-to-end KL against original Phi |
| runtime.py | chat loop: reference tokens -> watchdog -> spliced deliveries |
| finetune.py | protocol fine-tune: LoRA + memory-token deltas, masked loss, ref-decision eval |
| scripts/selftest_phi.py | real-model checks |
| scripts/make_corpus.py | PG19 books -> corpus.txt (train) + haystack.txt (held out) |
| scripts/eval_recall.py | planted-fact recall: original vs window-only vs tree |

## Known limits of this phase

- Training uses masked dense forms of the same math (O(T^2) FLOPs). Decoding
  reads only the window. Fast kernels come later.
- Padding is ignored: use unpadded or packed sequences.
- DynamicCache only (the cache stores RoPE'd and raw keys side by side).
- Synthetic fine-tune data teaches the protocol mechanics (keys like
  `peer0/k12`, values like `v123`). Real conversations make it natural.
- Needs a recent transformers (decoder passes `past_key_values`).
