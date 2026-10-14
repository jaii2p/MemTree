from memtree.runtime import (FILED_NOTE, WHISPER_HEADER, file_user_turn, render_memory,
                             split_for_window, whisper_query, window_budgets)
from memtree.store import LogGate, MemoryStore
from memtree.watchdog import Watchdog, _snippet


def _long(n_sent=60):
    return " ".join(f"Sentence {i} talks about lighthouses and boats and weather." for i in range(n_sent))


def test_split_keeps_short_messages_whole():
    assert split_for_window("Hi there. How are you?", 50, 20) == ([], "Hi there. How are you?")


def test_split_loses_nothing_and_respects_budgets():
    text = _long()
    spans, tail = split_for_window(text, tail_words=60, span_words=25)
    assert spans and len(tail.split()) <= 60
    assert all(len(s.split()) <= 25 for s in spans)
    assert " ".join(spans + [tail]).split() == text.split()  # every word kept, in order


def test_split_hard_splits_one_giant_sentence():
    text = " ".join(["word"] * 500) + ". Final short sentence."
    spans, tail = split_for_window(text, tail_words=10, span_words=50)
    assert tail == "Final short sentence."
    assert all(len(s.split()) <= 50 for s in spans) and sum(len(s.split()) for s in spans) == 500


def test_file_user_turn_copies_overflow_and_leaves_the_message_untouched():
    s = MemoryStore()
    g = LogGate(s)
    text = _long(40) + " The secret code for Harlan is 48213. " + _long(40) + " What is Harlan's code?"
    shown, qkey, tail = file_user_turn(g, "me", 7, text, tail_words=60, span_words=40)
    assert shown == text and qkey == "turn7:user" and tail.endswith("What is Harlan's code?")
    moved, _, _ = file_user_turn(MemoryStore and LogGate(MemoryStore()), "me", 1, text, 60, 40, move=True)
    assert moved.startswith(FILED_NOTE)
    b = s.branch("me")
    parts = [k for k in b.index if k.startswith("turn7:part")]
    assert parts and qkey in b.index
    assert any("48213" in b.get(k)[0].value for k in parts)  # the needle went out, intact
    s.check_invariants()


def test_whispers_surface_associations_with_a_snippet_around_the_match():
    s = MemoryStore()
    g = LogGate(s)
    text = _long(40) + " The secret code for Harlan is 48213. " + _long(40) + " What is Harlan's code?"
    shown, qkey, tail = file_user_turn(g, "me", 1, text, tail_words=60, span_words=40)
    dog = Watchdog(s)
    assert whisper_query(tail).endswith("What is Harlan's code?")
    ws = dog.whispers("me", whisper_query(tail, 6), exclude={qkey})
    assert ws and "48213" in ws[0].summary and ws[0].confidence == 0.5
    center = {n.key for n in dog.context_view("me")}
    assert all(w.key not in center for w in ws)
    block = render_memory(dog, "me", entry_chars=200, whispers=ws)
    assert WHISPER_HEADER in block and "48213" in block


def test_snippet_centres_on_the_match():
    text = "filler " * 200 + "Harlan has code 48213 " + "filler " * 200
    snip = _snippet(text, {"harl"}, 100)
    assert "Harlan" in snip and len(snip) <= 110 and snip.startswith("...")
    assert _snippet("short text", {"zzzz"}, 100) == "short text"


def test_render_memory_clips_fresh_entries():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "turn1:user", "x" * 1000)
    block = render_memory(Watchdog(s), "me", entry_chars=50)
    assert "x" * 50 + "..." in block and "x" * 60 not in block


def test_budgets_scale_with_window():
    a, b = window_budgets(1024), window_budgets(4096)
    assert a["tail_words"] < b["tail_words"] and a["entry_chars"] < b["entry_chars"]
