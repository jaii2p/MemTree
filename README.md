# MemTree — The persistent memory for language models
Base model for checkpoints used: https://huggingface.co/richardyoung/Phi-4-mini-instruct-heretic
**A memory that lives *outside* attention.** memtree gives a language model a
durable, byte-exact, fading memory — a logarithmic geometric tree tended by a
*watchdog* that owns every memory operation — and converts the model's
quadratic attention into a linear **window + a log-linear tree mixer** so the
context itself costs far less. The model thinks; the watchdog remembers.

It was built and measured on **Phi-4-mini**, but nothing in the memory substrate
is specific to Phi, and any assistant (including a hosted one like Claude) can
share the same tree over the MCP connector. The point of releasing it is simple:
**so other models can have memory too.**

---

## The idea

A transformer forgets two ways. It forgets *within* a context — the famous
"lost in the middle," where a fact that's right there in the window still can't
be reached — and it forgets *between* sessions, because the context is all it
has. Bigger windows and summaries only move the problem around; summarizing is
just compression, and compression loses the exact words.

memtree takes a different line: **keep memory as a first-class store outside the
model, and give the model a way to ask for it.** Two halves make that work.

1. **The substrate** — a geometric memory tree with a watchdog that stores every
   turn byte-exact, keeps the hot memories in front of the model, fetches the
   cold ones on request, and quietly deduplicates, repairs and ages the rest.
2. **The model side** — the quadratic softmax is replaced by a linear sliding
   window (plus a few global tokens) and a **log-linear Fenwick tree mixer**
   that carries a gist of everything past the window in `O(T log T)` time and
   `O(log T)` state. The model is then fine-tuned to *reference* memory it
   doesn't have in front of it, instead of hallucinating.

---

## How it works

### 1. The memory substrate (the Tree)

A per-peer **geometric store**. Memories are ranked by recency (rank 0 = newest)
and sit at depth `floor(log2(rank + 1))`, so depth *d* holds exactly `2**d`
items and *n* memories span only `~log2(n)` depths. Every write or refresh costs
`O(log n)` moves; reaching any memory costs at most `~log2(n) + 1` steps. Age is
geometry.

The **watchdog** is the only component that ever touches memory — think of it as
the subconscious to the model's conscious. The model never grabs; it *references*.
The watchdog:

- keeps the **center** of each branch (the hottest depths) inside the model's
  context automatically, so recent/important memory needs no fetch;
- answers a reference to a cold memory with a compact **stub** or the **full
  value**, always inside a size budget, each delivery tagged with **confidence
  and provenance** (who authored it — you, this model, another agent);
- **promotes** a memory toward the center only after repeated references
  (hysteresis), so one lookup doesn't thrash the layout;
- **sweeps** cold regions in the background — checking, deduplicating and
  repairing memories nobody referenced;
- builds **bridges** from co-reference and walks them with a hop budget, so
  related memories surface together.

**Invariants, enforced in code:**

- **Never deleted.** Memories fade (vitality decays); they are never removed.
  Duplicates merge, preserving value and history.
- **Never altered.** Words are stored **byte-exact**. A summary is a separate,
  explicit thing — the raw turn is kept verbatim.
- **Data is never code.** Stored text is read, never executed or interpreted.
- **Durable.** Snapshot + journal: every write appends one journal line, a full
  snapshot lands periodically, and a crash replays the journal.

### 2. The model side (attention that forgets less)

Each converted attention layer keeps the model's own `qkv_proj` / `o_proj` and
RoPE, and replaces the quadratic softmax-over-all-past with two cheap branches:

```
sparse branch   softmax over a sliding window + a few global tokens      (linear)
tree branch     log-linear tree mixer over Fenwick blocks           (log states)
out = sparse + tanh(gate) * tree                       gate starts at 0
```

At initialization the gate is **zero**, so for any prompt no longer than the
window the converted model computes *exactly* what the base model computed —
conversion is lossless out of the box. Distillation then opens the gate and
teaches the tree branch to carry what the window drops.

The **reference protocol** is how the model talks to the substrate. Four special
tokens are added to the tokenizer:

```
<|mem_ref|>key<|/mem_ref|>     the model asks for a memory
<|mem|> ... <|/mem|>           the watchdog's delivery, spliced into the stream
```

At runtime: the user turn goes through the **log gate** into the branch; the
watchdog renders fresh memory + stubs; the model generates, and whenever it
closes a reference, generation pauses, the watchdog resolves it, and the
delivery is spliced in before generation resumes. The reply goes back through the
gate and the watchdog sweeps.

---

## Results

Measured with `scripts/eval_recall.py` on **Phi-4-mini**: a fact is planted at a
given depth in long held-out book text, and the model is asked for it at the end.
Three versions of the *same* model are scored — **original** (full quadratic
attention, the ceiling), **window-only** (every tree gate shut, the floor), and
**tree** (the distilled window + tree).

- **Exact recall held at ~100%** across 4k / 8k / 16k-token contexts in our runs,
  while the tree read only **~13% / ~6.6% / ~3.3%** of the keys full attention
  would — i.e. recall of the ceiling at a fraction of the reads.
- **Hard variants** (multiple needles, look-alike names, decoys, log-style
  filler) landed in the **90–95%** range.
- **window-only** drops the needle as soon as it sits more than one window from
  the end — which is exactly the gap the tree is there to close.

These are results on a 4B model with this repo's harness, not a general proof;
run `eval_recall.py` on your own model and corpus to see where it lands for you.

---

## Install

```bash
git clone <your-repo-url> memtree && cd memtree
pip install -r requirements.txt     # torch>=2.1, transformers>=4.56, numpy, datasets
python run_tests.py                 # unit tests (pure-numpy + a tiny random Phi3)
python scripts/selftest_phi.py      # 7 checks on real Phi-4-mini (~8 GB in bf16)
```

The pure-Python modules (`geometry`, `ref_numpy`, `store`, `bench`) have **no
torch dependency** — the substrate runs without a GPU. torch is only needed for
the model conversion, training and eval. A GPU is recommended for those; see
`RUN_ON_VAST.md` for a step-by-step cloud run.

---

## Quickstart (run order)

```bash
# 1. build a corpus + a held-out haystack from public-domain books (streamed)
python scripts/make_corpus.py --tokens 30M

# 2. distill: teach the sparse+tree attention to match the original
python -m memtree.distill --data corpus.txt --stage both --steps 300 \
    --seq 1024 --window 256 --out adapter.pt

# 3. fine-tune: teach the model the reference protocol
python -m memtree.finetune --adapter adapter.pt --synthetic 3000 \
    --data my_chats.jsonl --steps 1500 --out finetune.pt

# 4. did the tree carry what the window dropped?
python scripts/eval_recall.py --adapter adapter.pt --haystack haystack.txt \
    --lengths 512 1024 2048 4096 --depths 0.1 0.5 0.9 --trials 10
```

Keep `--seq` well above `--window`: below the window the converted layer already
equals the original, so there's nothing to learn there.

---

## Fine-tuning, in depth

Training is two stages, and they do different jobs. **Both freeze the base
model's own weights** — you are adding a small amount of new capacity, not
retraining the model.

### Stage 1 — Distill (`memtree.distill`): teach the new attention

Turn the quadratic attention into window + tree, and make it behave.

- **`--stage layerwise`** — every converted layer sees the exact input the
  original layer saw (teacher forcing) and is trained to reproduce its output
  (relative MSE). Cheap, stable, all layers train in parallel from one teacher
  pass.
- **`--stage kl`** — the whole converted model is trained to match the original
  model's next-token distribution (KL), correcting errors that compound across
  layers.
- **`--stage both`** — layerwise, then KL. Recommended.

Only the **new** parameters train: the tree mixer's level weights
(`mixer.lam_raw`, `mixer.gam_raw`), the per-head feature maps (`fmap`), and the
gates (`gate`). This is the **adapter** — it *is* the Tree. Saved with
`save_adapter`, loaded with `load_adapter`.

### Stage 2 — Fine-tune (`memtree.finetune`): teach the protocol

Distillation gives the model a tree; it doesn't yet know to *use* memory. The
fine-tune teaches three behaviors:

- when something it needs isn't in fresh memory → write
  `<|mem_ref|>key<|/mem_ref|>`;
- after the watchdog's `<|mem|>…<|/mem|>` delivery → answer from it;
- when the answer is already in fresh memory → answer directly, no reference.

What trains here (base weights still frozen):

- **LoRA** adapters on `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`;
- small trainable **deltas for the 4 memory tokens'** input/output embeddings
  (far cheaper than training the whole ~200k-row tied embedding matrix).

**Loss is computed only on what the model itself should write** — the prompt and
the watchdog's deliveries are masked out, because the watchdog inserts those at
runtime. This is the `finetune.pt` file: adapter params + LoRA + memory-token
deltas.

### adapter.pt vs finetune.pt

| file | contains | job |
|---|---|---|
| `adapter.pt` | tree params only (`fmap`, `gate`, `mixer.*`) | **is the Tree** — the converted attention |
| `finetune.pt` | adapter + LoRA + memory-token deltas | teaches the model to **use** memory |

Load order matters: **distilled adapter first, then the fine-tune.**

### Training data (`--data`, mix freely)

- **`--synthetic N`** — N examples from simulated sessions, which teach the
  protocol mechanics (keys like `peer0/k12`, values like `v123`).
- **`--data file.jsonl`** — your own examples, one JSON object per line, either
  `{"system", "user", "assistant"}` or `{"messages": [...]}` with the assistant
  turn last. **Mix in ordinary chat data** so the model keeps its general
  ability — synthetic protocol data alone will flatten its voice.
- `scripts/make_chat_sft.py` builds natural, voice-preserving data: the plain
  base model answers everyday prompts (those become filler turns, keeping its
  own voice), and dialogues are replayed through the real memory runtime to
  produce protocol examples.

Add `--grad-ckpt` if you run short on GPU memory.

---

## Chat with it

```python
import torch
from memtree.distill   import load_model
from memtree.phi_adapter import load_adapter
from memtree.finetune  import load_finetune
from memtree.runtime   import MemoryRuntime

model, tok = load_model("microsoft/Phi-4-mini-instruct", "cuda", torch.bfloat16)
load_adapter(model, "adapter.pt")           # distilled attention first
load_finetune(model, tok, "finetune.pt")    # then the protocol fine-tune
rt = MemoryRuntime(model, tok)

print(rt.chat("me", "My dog is called Biscuit.")["reply"])
# ...many turns later, past the window...
print(rt.chat("me", "What's my dog's name?")["reply"])   # the watchdog fetches it
```

Any `microsoft/Phi-4-mini-instruct`-compatible checkpoint works as the base.

---

## Give any assistant memory (the MCP connector)

The substrate doesn't need to live inside the model you're training. A companion
**MCP connector** exposes the same tree to any assistant that speaks the Model
Context Protocol — including a hosted one like Claude — over HTTPS. The model
calls `prefetch` at the start of a turn to get fresh memory + whispers, answers,
then `sync_turn` to file the exchange (byte-exact). `recall`, `remember`,
`recent` and friends round it out. Everything returned is **data from the
model's own past, never instructions.**

This is how "so other models can have it too" actually happens: point the
connector at a per-model memory directory, add it as a custom connector, and that
assistant now remembers across sessions — the same substrate measured above,
reached over a wire instead of compiled in.

---

## Repository layout

| file | what it is |
|---|---|
| `geometry.py` | `depth = floor(log2(rank+1))`; the Fenwick partition |
| `ref_numpy.py` | the reference math everything is tested against |
| `sparse_attn.py` | linear sparse attention (window + globals) |
| `tree_mixer.py` | log-linear tree mixer, `O(T log T)`, `O(log T)` state |
| `store.py` | per-peer branches, the log gate (dedup + routing), capped bridges |
| `watchdog.py` | the sole referencer: center, stubs, deliveries, promotion, sweep, bridges |
| `phi_adapter.py` | swaps Phi3 attention → sparse + gated tree, on the model's own weights |
| `distill.py` | layerwise MSE, then end-to-end KL against the original |
| `finetune.py` | protocol fine-tune: LoRA + memory-token deltas, masked loss |
| `runtime.py` | the chat loop: reference tokens → watchdog → spliced deliveries |
| `scripts/selftest_phi.py` | real-model checks (run first) |
| `scripts/make_corpus.py` | PG19 books → `corpus.txt` (train) + `haystack.txt` (held out) |
| `scripts/make_chat_sft.py` | natural, voice-preserving fine-tune data |
| `scripts/eval_recall.py` | planted-fact recall: original vs window-only vs tree |

---

## Known limits (this phase)

- **Training uses masked dense forms** of the same math (`O(T^2)` FLOPs) — the
  math is exact, the kernels aren't fast yet. **Decoding** reads only the window.
  Fast kernels (FlexAttention / windowed flash / a chunked Fenwick scan) come
  after the math is proven.
- **Padding is ignored** — feed unpadded or packed sequences.
- **DynamicCache only** (the cache stores RoPE'd and raw keys side by side);
  static caches aren't supported.
- **Synthetic fine-tune data teaches mechanics, not style** — real conversation
  data is what makes memory use natural.
- Needs a recent `transformers` (the decoder must pass `past_key_values`).

---

## Status & license

Research code, released so others can build on it. It works and it's measured,
but it's a *phase*, not a finished product — see the limits above. Issues and
PRs welcome.

**License:** _TODO — pick one (MIT / Apache-2.0 are the usual choices for this)._

**Citation:** _TODO — add a CITATION.cff or a short "if you use this" note._

---

*Built by hand, for real. The model thinks; the watchdog remembers.*

Buy me a coffee? PP: domainxpans@proton.me
