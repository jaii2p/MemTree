import math
import random

import numpy as np

from memtree.geometry import levels_needed
from memtree.store import BridgeTable, LogGate, MemoryStore, Node


def _node(key, peer="p"):
    return Node(key=key, peer=peer, value=key)


def test_perfect_geometry_for_full_trees():
    from memtree.store import Branch
    for k in range(1, 8):
        b = Branch("p")
        for i in range((1 << k) - 1):
            b.insert_front(_node(f"n{i}"))
        assert [len(l) for l in b.levels] == [1 << d for d in range(k)]
        b.check_invariants()


def test_invariants_hold_under_random_inserts_touches_removes():
    from memtree.store import Branch
    rng = random.Random(3)
    b = Branch("p")
    live = []
    for i in range(600):
        op = rng.random()
        if op < 0.55 or not live:
            b.insert_front(_node(f"n{i}"))
            live.append(f"n{i}")
        elif op < 0.85:
            b.touch(rng.choice(live))
        else:
            k = rng.choice(live)
            b.remove(k)
            live.remove(k)
        b.check_invariants()
        assert len(b) == len(live)


def test_push_and_touch_cost_is_logarithmic():
    from memtree.store import Branch
    b = Branch("p")
    N = 2000
    for i in range(N):
        before = b.ops
        b.insert_front(_node(f"n{i}"))
        assert b.ops - before <= max(1, levels_needed(len(b)))
    assert b.ops <= N * (math.log2(N) + 1)
    keys = list(b.index)
    for key in random.Random(1).sample(keys, 200):
        before = b.ops
        b.touch(key)
        assert b.ops - before <= 2 * levels_needed(len(b))
    b.check_invariants()


def test_reaching_any_node_takes_at_most_log_steps():
    from memtree.store import Branch
    b = Branch("p")
    for i in range(1000):
        b.insert_front(_node(f"n{i}"))
    bound = levels_needed(len(b))
    assert max(d + 1 for d in b.index.values()) == bound <= math.log2(1000) + 1


def test_bridge_degree_cap_symmetry_and_weakest_eviction():
    t = BridgeTable(max_degree=2)
    a, b, c, d = [("p", x) for x in "abcd"]
    t.add(a, b, 1.0)
    t.add(a, c, 2.0)
    t.add(a, d, 3.0)
    assert set(t.neighbors(a)) == {c, d}
    assert a not in t.neighbors(b)
    t.check_invariants()
    rng = random.Random(0)
    big = BridgeTable(max_degree=3)
    refs = [("p", str(i)) for i in range(40)]
    for _ in range(500):
        big.add(rng.choice(refs), rng.choice(refs), rng.random())
        big.check_invariants()


def test_gate_dedup_update_provenance_and_cross_peer_bridge():
    s = MemoryStore()
    g = LogGate(s)
    assert g.ingest("a", "color", "red") == "new"
    assert g.ingest("a", "color", "red") == "refresh"
    assert g.ingest("a", "color", "blue") == "update"
    node, depth = s.lookup("a", "color")
    assert node.value == "blue" and node.history == ["red"] and len(node.mentions) == 3
    assert len(s.branch("a")) == 1 and depth == 0
    assert g.ingest("b", "color", "green") == "new"  # shared fact -> bridge, not a copy
    assert ("b", "color") in s.bridges.neighbors(("a", "color"))
    s.check_invariants()


def test_gate_near_duplicate_merges_shallow_embeddings():
    s = MemoryStore()
    g = LogGate(s, sim_threshold=0.95)
    e = np.array([1.0, 0.0, 0.0])
    assert g.ingest("a", "x", "v", emb=e) == "new"
    assert g.ingest("a", "x2", "v", emb=e * 1.0001 + 1e-4) == "near_dup"
    assert g.ingest("a", "y", "w", emb=np.array([0.0, 1.0, 0.0])) == "new"
    node, _ = s.lookup("a", "x")
    assert node.aliases == ["x2"] and s.branch("a").aliases["x2"] == "x"
    assert len(s.branch("a")) == 2
    s.check_invariants()
