"""Build the distillation corpus from PG19 (public-domain books, long-range text).

    pip install datasets
    python scripts/make_corpus.py --tokens 30M

Writes:
  corpus.txt     training text from the PG19 train split (~4 characters per token)
  haystack.txt   held-out text from the validation split, used by eval_recall.py
                 as filler around planted facts; never trained on

Books are streamed, so nothing near the full 7 GB dataset is downloaded.
"""
import argparse
import os
import re
import sys
import time

CHARS_PER_TOKEN = 4.0  # English with Phi-4's tokenizer, roughly


def parse_count(s: str) -> int:
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMbB]?)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad count {s!r}, use e.g. 500k, 30M")
    mult = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9}[m.group(2).lower()]
    return int(float(m.group(1)) * mult)


def clean(text: str) -> str:
    """Join hard-wrapped lines into paragraphs, drop runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p)


def book_text(row) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    strings = [v for v in row.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else ""


def write_split(name: str, split: str, out_path: str, target_chars: int, min_chars: int) -> int:
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True)
    written = books = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = clean(book_text(row))
            if len(text) < min_chars:
                continue
            f.write(text)
            f.write("\n\n")
            written += len(text) + 2
            books += 1
            if books % 25 == 0:
                print(f"  {out_path}: {books} books, ~{written / CHARS_PER_TOKEN / 1e6:.1f}M tokens "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if written >= target_chars:
                break
    if written < target_chars:
        print(f"  warning: split ran out at ~{written / CHARS_PER_TOKEN / 1e6:.1f}M tokens")
    print(f"wrote {out_path}: {books} books, {written / 1e6:.0f}M chars "
          f"(~{written / CHARS_PER_TOKEN / 1e6:.1f}M tokens)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=parse_count, default=parse_count("30M"))
    ap.add_argument("--haystack-tokens", type=parse_count, default=parse_count("500k"))
    ap.add_argument("--dataset", default="fla-hub/pg19")
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--haystack-out", default="haystack.txt")
    ap.add_argument("--min-chars", type=int, default=50_000, help="skip books shorter than this")
    a = ap.parse_args(argv)
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("needs the datasets library: pip install datasets")
    write_split(a.dataset, "train", a.out, int(a.tokens * CHARS_PER_TOKEN), a.min_chars)
    write_split(a.dataset, "validation", a.haystack_out,
                int(a.haystack_tokens * CHARS_PER_TOKEN), a.min_chars)
    print(f"done. corpus {os.path.getsize(a.out) / 1e6:.0f} MB, "
          f"haystack {os.path.getsize(a.haystack_out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
