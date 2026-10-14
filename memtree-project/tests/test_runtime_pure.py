from memtree.bench import make_session
from memtree.runtime import (MEM, MEM_END, MEM_REF, MEM_REF_END, reference_sft_examples,
                             render_delivery, render_memory)
from memtree.store import LogGate, MemoryStore
from memtree.watchdog import Watchdog


def test_render_memory_shows_center_and_stubs():
    s = MemoryStore()
    g = LogGate(s)
    for i in range(10):
        g.ingest("a", f"k{i}", f"value {i} " * 5)
    dog = Watchdog(s)
    text = render_memory(dog, "a")
    assert text.startswith("Fresh memory:")
    assert "[k9]" in text and "[k8]" in text and "[k7]" in text  # center
    assert MEM_REF in text and "[k6]" in text  # first outer depth as stubs
    assert "[k0]" not in text  # deep: not shown unless bridged


def test_render_delivery_forms():
    s = MemoryStore()
    g = LogGate(s)
    for i in range(10):
        g.ingest("a", f"k{i}", f"v{i}")
    dog = Watchdog(s)
    assert render_delivery(dog.reference("a", "k9")).endswith("already in fresh memory" + MEM_END)
    assert render_delivery(dog.reference("a", "k1", expand=True)) == f"{MEM}[k1] v1{MEM_END}"
    assert "not found" in render_delivery(dog.reference("a", "zz"))


def test_sft_examples_reference_outer_and_answer_center_directly():
    ex = reference_sft_examples(make_session(seed=0, n_events=600))
    assert ex
    outer = [e for e in ex if e["assistant"].startswith(MEM_REF)]
    direct = [e for e in ex if not e["assistant"].startswith(MEM_REF)]
    assert outer and direct
    for e in outer:
        key = e["user"][len("What is "):-1]
        assert e["assistant"].startswith(f"{MEM_REF}{key}{MEM_REF_END}{MEM}[{key}]")


def test_clean_reply_strips_references_deliveries_and_specials():
    from memtree.runtime import clean_reply
    raw = f"Sure. {MEM_REF}k1{MEM_REF_END}{MEM}[k1] v1{MEM_END}It is v1.<|end|>"
    assert clean_reply(raw) == "Sure. It is v1."


def test_delivery_shows_matched_key_for_search_hits():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "turn1:user", "My dog is Biscuit.")
    for i in range(2, 12):
        g.ingest("me", f"turn{i}:user", f"chatter {i}")
    dog = Watchdog(s)
    text = render_delivery(dog.reference("me", "dog", expand=True))
    assert text == f"{MEM}[turn1:user] My dog is Biscuit.{MEM_END}"
