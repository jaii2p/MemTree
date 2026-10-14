"""Build natural fine-tune data in Phi's own voice.

    python scripts/make_chat_sft.py --model richardyoung/Phi-4-mini-instruct-heretic --out chat_sft.jsonl

1. Original Phi (plain model, original attention, no adapter) answers a few
   hundred everyday prompts. Those answers are the filler turns, so the
   fine-tune keeps her own voice instead of overwriting it.
2. Dialogues (facts, filler, natural questions) are replayed through the real
   memory runtime to produce training examples (memtree/chat_data.py).

Use --no-generate to skip step 1 and use a small built-in filler set (fast, but
less voice-preserving).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memtree.chat_data import CANNED_FILLER, build_dataset  # noqa: E402

TOPICS = ["cooking", "space", "history", "dogs", "cars", "music", "the ocean", "sleep", "exercise",
          "budgeting", "gardening", "weather", "computers", "movies", "coffee", "travel", "insects",
          "volcanoes", "electricity", "chess", "photography", "birds", "camping", "bread", "the brain"]
FORMS = ["Tell me a fun fact about {t}.", "Give me a quick tip about {t}.", "Explain something interesting about {t} in two sentences.",
         "What's a common myth about {t}?", "Ask me a question about {t}.", "What's one thing beginners get wrong about {t}?"]
EXTRA = ["How do I boil an egg?", "Why is the sky blue?", "How far is the moon?", "Give me a one-line joke.",
         "What's a good way to cook rice?", "Explain how a rainbow forms.", "How do vaccines work, briefly?",
         "Write a two-line poem about rain.", "What's 15% of 80?", "How do I get better at waking up early?",
         "Suggest a name for a goldfish.", "What should I pack for a weekend trip?", "How does a microwave heat food?",
         "Give me three ideas for a rainy day.", "What's the difference between weather and climate?",
         "How do I fix a squeaky door?", "Recommend a simple stretching routine.", "Why do cats purr?",
         "How many continents are there?", "Explain compound interest simply.", "hey how's it going?",
         "I'm bored, entertain me.", "What's a good habit to start this week?", "Summarize what a black hole is."]


def prompts(n: int, seed: int):
    rng = random.Random(seed)
    pool = [f.replace("{t}", t) for t in TOPICS for f in FORMS] + EXTRA
    rng.shuffle(pool)
    return pool[:n]


def generate_answers(model_name, prompt_list, device, batch=32, max_new_tokens=200):
    import torch
    from memtree.distill import load_model
    model, tok = load_model(model_name, device, torch.bfloat16)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    t0 = time.time()
    for i in range(0, len(prompt_list), batch):
        chunk = prompt_list[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7,
                               top_p=0.9, pad_token_id=tok.pad_token_id)
        for p, row in zip(chunk, g[:, enc["input_ids"].shape[1]:]):
            a = tok.decode(row, skip_special_tokens=True).strip()
            if len(a) > 10:
                out.append((p, a))
        print(f"  answered {min(i + batch, len(prompt_list))}/{len(prompt_list)} ({time.time() - t0:.0f}s)", flush=True)
    del model
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--prompts", type=int, default=160)
    ap.add_argument("--dialogues", type=int, default=1200)
    ap.add_argument("--out", default="chat_sft.jsonl")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--filler-from", default=None,
                    help="reuse answers from an earlier run's .filler.json (skips generation)")
    ap.add_argument("--passages", default="corpus.txt",
                    help="training text for long-passage dialogues (never haystack.txt, the eval text)")
    ap.add_argument("--n-passages", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.filler_from:
        with open(a.filler_from) as f:
            filler = [tuple(x) for x in json.load(f)]
    elif a.no_generate:
        filler = list(CANNED_FILLER)
    else:
        filler = generate_answers(a.model, prompts(a.prompts, a.seed), a.device)
        with open(a.out + ".filler.json", "w") as f:
            json.dump(filler, f)
    passages = None
    if a.passages and os.path.exists(a.passages):
        with open(a.passages, encoding="utf-8") as f:
            text = f.read(30_000_000)
        words = text.split()
        chunks = [" ".join(words[i:i + 3000]) for i in range(0, len(words) - 3000, 3000)]
        random.Random(a.seed).shuffle(chunks)
        passages = chunks[:a.n_passages]
        print(f"long-passage dialogues use {len(passages)} 3000-word chunks from {a.passages}")
    else:
        print("no passage text found; long-passage dialogues use filler answers as text")
    examples, stats = build_dataset(a.dialogues, filler, seed=a.seed, passages=passages)
    with open(a.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(examples)} examples to {a.out}: {stats}")


if __name__ == "__main__":
    main()
