from memtree.store import HALF_LIFE, LogGate, MemoryStore, strengthen, vitality
from memtree.watchdog import Watchdog


def test_memories_fade_when_left_alone_and_never_vanish():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "old", "the lighthouse keeper's cat is Juniper")
    b = s.branch("me")
    node = b.get("old")[0]
    v0 = vitality(node, b.clock)
    for i in range(HALF_LIFE):
        g.ingest("me", f"chat{i}", f"filler number {i}")
    v1 = vitality(node, b.clock)
    assert 0.45 < v1 / v0 < 0.55  # one half-life later: half as alive
    assert "old" in b.index  # faint, not gone
    hit = Watchdog(s).reference("me", "Juniper cat", expand=True)
    assert hit.found and "Juniper" in hit.value  # a strong cue still reaches it


def test_recall_and_repetition_make_memories_stronger():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "k", "boat is red")
    b = s.branch("me")
    node = b.get("k")[0]
    before = vitality(node, b.clock)
    Watchdog(s).reference("me", "k")
    after_ref = vitality(node, b.clock)
    g.ingest("me", "k", "boat is red")  # met again
    after_again = vitality(node, b.clock)
    assert before < after_ref < after_again


def test_stronger_memories_come_forward_for_the_same_cue():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "a", "my favorite tea is jasmine")
    g.ingest("me", "b", "my favorite tea is oolong")
    dog = Watchdog(s)
    b = s.branch("me")
    strengthen(b.get("a")[0], b.clock, 3.0)  # 'a' has been recalled a lot
    top = dog.search("me", "favorite tea", limit=1)[0][0]
    assert top.key == "a"


def test_whispers_need_a_real_match_and_strengthen_what_they_surface():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "fact", "The secret code for Harlan is 48213.")
    for i in range(10):
        g.ingest("me", f"f{i}", f"weather chatter {i}")
    dog = Watchdog(s)
    b = s.branch("me")
    node = b.get("fact")[0]
    v = vitality(node, b.clock)
    ws = dog.whispers("me", "what was the code for Harlan")
    assert ws and ws[0].key == "fact" and vitality(node, b.clock) > v
    assert dog.whispers("me", "completely unrelated volcano") == []
