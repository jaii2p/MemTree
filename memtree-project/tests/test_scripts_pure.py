import importlib.util
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_insert_needle_lands_on_sentence_boundary_at_depth():
    ev = _load("eval_recall")
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    for depth in (0.0, 0.1, 0.5, 0.9, 1.0):
        out = ev.insert_needle(text, "NEEDLE.", depth)
        i = out.index("NEEDLE.")
        assert out.replace("NEEDLE. ", "", 1) == text
        assert abs(i / len(text) - depth) < 0.05
        assert i == 0 or out[i - 2] == "." or i == len(text)


def test_make_trial_and_scoring():
    ev = _load("eval_recall")
    tr = ev.make_trial(random.Random(0), "Some filler. More filler. End.", 0.5)
    assert tr["needle"] in tr["doc"] and tr["answer"] in tr["needle"] and len(tr["answer"]) == 5
    assert ev.is_correct(f" {tr['answer'][:2]},{tr['answer'][2:]}\n", tr["answer"])
    assert not ev.is_correct("12345" if tr["answer"] != "12345" else "54321", tr["answer"])


def test_attention_reads_match_reference_counts():
    ev = _load("eval_recall")
    from memtree.ref_numpy import sparse_indices
    for T, w, g in [(100, 16, 4), (600, 256, 4), (5, 8, 2)]:
        r = ev.attention_reads(T, w, g)
        assert r["window-only"] == int((sparse_indices(T, w, g) >= 0).sum())
        assert r["original"] == T * (T + 1) // 2
        assert r["window-only"] < r["tree"] < r["original"] or T < 20
    r = ev.attention_reads(4096, 256, 4)
    assert r["tree"] / r["original"] < 0.15


def test_make_corpus_helpers():
    mc = _load("make_corpus")
    assert mc.parse_count("30M") == 30_000_000 and mc.parse_count("500k") == 500_000
    assert mc.parse_count("1.5m") == 1_500_000
    raw = "Line one of a\nwrapped paragraph.\r\n\r\n\n\nSecond   para\nhere.\n"
    assert mc.clean(raw) == "Line one of a wrapped paragraph.\n\nSecond   para here."
    assert mc.book_text({"text": "abc", "url": "x"}) == "abc"
    assert mc.book_text({"title": "t", "body": "longer body"}) == "longer body"
