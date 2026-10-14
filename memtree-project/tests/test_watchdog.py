from memtree.store import LogGate, MemoryStore
from memtree.watchdog import Watchdog


def _filled(n, peer="a", prefix="k", **kw):
    s = MemoryStore()
    g = LogGate(s)
    for i in range(n):
        g.ingest(peer, f"{prefix}{i}", f"value-number-{i}-with-some-length")
    return s, g, Watchdog(s, **kw)


def test_center_is_kept_in_context_and_main_never_grabs():
    s, g, dog = _filled(15)
    b = s.branch("a")
    view = [n.key for n in dog.context_view("a")]
    assert view == ["k14", "k13", "k12"]  # depth 0 + depth 1, newest first
    assert len(view) == (1 << dog.boundary) - 1
    assert not hasattr(dog, "main_read")
    assert len(dog.boundary_stubs("a")) == 4  # first outer depth
    assert b.index["k5"] == 3


def test_reference_to_center_confirms_refreshes_and_delivers_nothing():
    s, g, dog = _filled(15)
    b = s.branch("a")
    assert b.index["k13"] == 1
    d = dog.reference("a", "k13")
    assert d.found and d.resolution == "in_context" and d.value is None and d.hops == 2
    assert b.index["k13"] == 0  # hot memory pulled to the very center
    d = dog.reference("a", "k13")
    assert d.resolution == "in_context" and d.hops == 1 and b.index["k13"] == 0
    b.check_invariants()


def test_proactive_surfaces_bridged_outer_memories():
    s, g, dog = _filled(15)
    s.bridges.add(("a", "k14"), ("a", "k3"))  # center <-> deep
    s.bridges.add(("a", "k14"), ("a", "k13"))  # center <-> center: not surfaced
    stubs = dog.proactive("a")
    assert [st.key for st in stubs] == ["k3"] and stubs[0].depth == 3
    assert stubs[0].confidence == 0.5


def test_cold_deep_memory_returns_stub_then_full_when_hot_or_expanded():
    s, g, dog = _filled(15)
    d = dog.reference("a", "k5")
    assert d.found and d.resolution == "stub" and d.value is None
    assert d.stub.summary.endswith("...") and d.confidence == 1.0 and d.depth == 3
    assert dog.reference("a", "k6", expand=True).resolution == "full"
    dog.reference("a", "k7")
    dog.reference("a", "k7")
    assert dog.reference("a", "k7").resolution == "full"  # 3rd ref: hot


def test_budget_limits_delivery_size():
    s, g, dog = _filled(15)
    d = dog.reference("a", "k5", budget_chars=5)
    assert d.resolution == "stub" and len(d.stub.summary) <= 5
    d = dog.reference("a", "k6", budget_chars=5, expand=True)
    assert d.resolution == "stub"  # full value does not fit the budget


def test_promotion_needs_repeated_references():
    s, g, dog = _filled(15, promote_votes=3, hot_refs=99)
    b = s.branch("a")
    assert b.index["k5"] == 3
    dog.reference("a", "k5")
    dog.reference("a", "k5")
    assert b.index["k5"] == 3  # two votes: no move yet (hysteresis)
    d = dog.reference("a", "k5")
    assert d.promoted and b.index["k5"] == 0
    assert b.get("k5")[0].votes == 0
    b.check_invariants()


def test_miss_and_alias_resolution():
    s, g, dog = _filled(15)
    d = dog.reference("a", "nope")
    assert not d.found and d.resolution == "miss" and d.hops == len(s.branch("a").levels)
    assert not dog.reference("zzz", "k1").found


def test_sweep_merges_normalized_duplicates_in_cold_regions():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("a", "Foo Bar", "x")
    g.ingest("a", "foo-bar", "y")
    for i in range(30):
        g.ingest("a", f"filler{i}", i)
    dog = Watchdog(s)
    b = s.branch("a")
    assert b.index["Foo Bar"] >= 2 and b.index["foo-bar"] >= 2
    n_before = len(b)
    total = {"merged": 0}
    for _ in range(20):
        total["merged"] += dog.sweep("a", budget=5)["merged"]
    assert total["merged"] == 1 and len(b) == n_before - 1
    assert dog.reference("a", "foo-bar", expand=True).found  # alias still resolves
    b.check_invariants()
    s.check_invariants()


def test_sweep_visits_every_deep_node_and_decays_votes():
    s, g, dog = _filled(63)
    b = s.branch("a")
    deep = [n for n in b.nodes_by_recency() if b.index[n.key] >= 2]
    for n in deep:
        n.votes = 2
    for _ in range(len(deep)):  # budget 3 per call: more than enough calls
        dog.sweep("a", budget=3)
    assert all(n.votes <= 1 for n in deep)
    assert all(n.votes == 0 for n in b.nodes_by_recency() if b.index[n.key] < 2)


def test_bridge_walk_handles_cycles_and_hop_budget():
    s, g, dog = _filled(6)
    A, B, C, D = [("a", f"k{i}") for i in range(4)]
    for x, y in [(A, B), (B, C), (C, A), (C, D)]:
        s.bridges.add(x, y)
    rel = dog.related("a", "k0", max_hops=3)
    refs = [r for r, _, _ in rel]
    assert len(refs) == len(set(refs)) == 3 and A not in refs
    assert {r for r, h, _ in dog.related("a", "k0", max_hops=1)} == {B, C}


def test_co_reference_creates_bridge_and_visibility_is_enforced():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("a", "x", 1)
    g.ingest("b", "y", 2)
    g.ingest("b", "z", 3, visibility="shared")
    dog = Watchdog(s, allow_cross_peer=False)
    dog.reference("a", "x", co_refs=(("b", "y"), ("b", "z")))
    assert ("b", "y") not in s.bridges.neighbors(("a", "x"))  # private: blocked
    assert ("b", "z") in s.bridges.neighbors(("a", "x"))  # shared: allowed
    dog2 = Watchdog(s, allow_cross_peer=True)
    dog2.reference("a", "x", co_refs=(("b", "y"),))
    assert ("b", "y") in s.bridges.neighbors(("a", "x"))


def _chat_store():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "turn1:user", "My dog is called Biscuit and she hates thunder.")
    g.ingest("me", "turn1:assistant", "Biscuit sounds lovely! Thunder is scary for dogs.")
    g.ingest("me", "turn2:user", "I work as a welder in Tulsa.")
    for i in range(3, 20):
        g.ingest("me", f"turn{i}:user", f"Random chatter number {i} about the weather.")
    return s, g


def test_search_finds_memory_by_content_when_key_misses():
    s, g = _chat_store()
    dog = Watchdog(s)
    assert s.branch("me").index["turn1:user"] >= 2  # deep: not in Phi's context
    d = dog.reference("me", "dog name", expand=True)
    assert d.found and d.provenance["matched_by"] == "search" and d.confidence == 0.6
    assert d.provenance["key"] in ("turn1:user", "turn1:assistant") and "Biscuit" in d.value
    d = dog.reference("me", "where do I work", expand=True)
    assert d.found and d.provenance["key"] == "turn2:user" and "welder" in d.value
    assert not dog.reference("me", "favourite volcano").found  # no shared words: real miss


def test_search_prefers_rarer_words_and_respects_exclude():
    s, g = _chat_store()
    dog = Watchdog(s)
    top = dog.search("me", "thunder weather", limit=1)[0][0]
    assert top.key in ("turn1:user", "turn1:assistant")  # 'thunder' rare, 'weather' common
    g.ingest("me", "turn20:user", "What's my dog called?")
    d = dog.reference("me", "my dog called", expand=True, exclude={"turn20:user"})
    assert d.provenance["key"] != "turn20:user" and "Biscuit" in d.value


def test_search_survives_merges_and_updates():
    s = MemoryStore()
    g = LogGate(s)
    g.ingest("me", "fact", "the boat is red")
    g.ingest("me", "fact", "the boat is blue")  # update
    dog = Watchdog(s)
    assert dog.search("me", "red")  == []  # stale word no longer matches
    assert dog.search("me", "blue boat")[0][0].key == "fact"
    s.check_invariants()
