"""Geometric memory store: per-peer branches, depth = age, capacity doubling.

Invariant (checked by Branch.check_invariants): items ordered by recency
(rank 0 = newest) sit at depth floor(log2(rank + 1)); depths 0..k-1 are exactly
full (2**d items) and the last depth is partial. Every push/refresh costs
O(log n) level moves; reaching depth d costs d + 1 <= log2(n) + 1 steps.
"""
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import depth_of_rank, levels_needed

Ref = Tuple[str, str]  # (peer, key)

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""a an and are as at be but by do does did for from had has have he her
him his how i if in into is it its me my of on or our she so than that the their them then
there they this to too us was we were what whats when where which who whom why will with
you your yours am can could would should just about tell remember called turn user assistant
""".split())


def words_of(text) -> set:
    """Content words of a memory or a query: lowercase alphanumerics, no
    stopwords, cut to a 4-letter stem so allergy/allergic and drive/driving
    match."""
    return {w[:4] for w in _WORD.findall(str(text).lower()) if len(w) > 1 and w not in STOPWORDS}


@dataclass
class Node:
    key: str
    peer: str
    value: Any
    emb: Optional[np.ndarray] = None
    visibility: str = "private"  # "private" | "shared"
    created: int = 0
    touched: int = 0
    mentions: List[int] = field(default_factory=list)  # provenance: when seen
    history: List[Any] = field(default_factory=list)  # superseded values
    aliases: List[str] = field(default_factory=list)  # merged near-duplicate keys
    refs: int = 0  # times referenced by the main model
    votes: int = 0  # pending promotion votes (hysteresis)
    strength: float = 1.0  # how alive the memory is, as of tick `s_clock`
    s_clock: int = 0


HALF_LIFE = 60  # ticks (about 20 chat turns) for an untouched memory to fade by half


def vitality(node: Node, clock: int, half_life: float = HALF_LIFE) -> float:
    """Memories are faint living things: strength fades with time, computed
    lazily so nothing has to visit every memory to age it."""
    return node.strength * 0.5 ** (max(0, clock - node.s_clock) / half_life)


def strengthen(node: Node, clock: int, amount: float = 1.0) -> float:
    """Recalling or meeting a memory again makes it stronger. Never deletes."""
    node.strength = vitality(node, clock) + amount
    node.s_clock = clock
    return node.strength


class Branch:
    """One peer's tree. levels[d] is an OrderedDict key -> Node, oldest first."""

    def __init__(self, peer: str):
        self.peer = peer
        self.levels: List[OrderedDict] = [OrderedDict()]
        self.index: Dict[str, int] = {}  # key -> depth
        self.aliases: Dict[str, str] = {}  # merged duplicate key -> canonical key
        self.clock = 0
        self.ops = 0  # level moves, for cost accounting

    def __len__(self):
        return len(self.index)

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def insert_front(self, node: Node) -> None:
        """Place at depth 0; overflow cascades the oldest of each full level
        down one depth."""
        node.touched = self.tick()
        self.levels[0][node.key] = node
        self.index[node.key] = 0
        i = 0
        while len(self.levels[i]) > (1 << i):
            k, n = self.levels[i].popitem(last=False)
            if i + 1 == len(self.levels):
                self.levels.append(OrderedDict())
            self.levels[i + 1][k] = n
            self.index[k] = i + 1
            self.ops += 1
            i += 1

    def remove(self, key: str) -> Node:
        """Remove and re-pack: each deeper level hands its newest item up."""
        j = self.index.pop(key)
        node = self.levels[j].pop(key)
        for i in range(j, len(self.levels) - 1):
            if not self.levels[i + 1]:
                break
            k, n = self.levels[i + 1].popitem(last=True)
            self.levels[i][k] = n
            self.levels[i].move_to_end(k, last=False)
            self.index[k] = i
            self.ops += 1
        while len(self.levels) > 1 and not self.levels[-1]:
            self.levels.pop()
        return node

    def touch(self, key: str) -> Node:
        """Pull an existing node back to depth 0 (refresh / promotion)."""
        node = self.remove(key)
        self.insert_front(node)
        return node

    def get(self, key: str):
        d = self.index.get(key)
        if d is None:
            return None
        return self.levels[d][key], d

    def nodes_by_recency(self):
        for level in self.levels:
            yield from reversed(level.values())

    def depth_nodes(self, depth: int):
        if depth >= len(self.levels):
            return []
        return list(reversed(self.levels[depth].values()))

    def check_invariants(self) -> None:
        n = len(self.index)
        assert n == sum(len(l) for l in self.levels), "index/levels size mismatch"
        if n == 0:
            assert len(self.levels) == 1
            return
        assert len(self.levels) == levels_needed(n), "wrong number of depths"
        for i, l in enumerate(self.levels[:-1]):
            assert len(l) == 1 << i, f"depth {i} not exactly full"
        assert 1 <= len(self.levels[-1]) <= 1 << (len(self.levels) - 1)
        last_touched = None
        for rank, node in enumerate(self.nodes_by_recency()):
            assert self.index[node.key] == depth_of_rank(rank), "wrong depth for rank"
            assert last_touched is None or node.touched < last_touched, "order broken"
            last_touched = node.touched


class BridgeTable:
    """Symmetric cross-references with a hard degree cap (weakest evicted)."""

    def __init__(self, max_degree: int = 4):
        self.max_degree = max_degree
        self.adj: Dict[Ref, Dict[Ref, float]] = {}

    def neighbors(self, a: Ref) -> Dict[Ref, float]:
        return self.adj.get(a, {})

    def add(self, a: Ref, b: Ref, strength: float = 1.0) -> None:
        if a == b:
            return
        s = self.adj.setdefault(a, {}).get(b, 0.0) + strength
        self.adj[a][b] = s
        self.adj.setdefault(b, {})[a] = s
        self._enforce(a)
        self._enforce(b)

    def _enforce(self, x: Ref) -> None:
        nb = self.adj.get(x)
        while nb and len(nb) > self.max_degree:
            y = min(nb, key=lambda n: (nb[n], repr(n)))  # weakest, deterministic ties
            self._drop(x, y)

    def _drop(self, x: Ref, y: Ref) -> None:
        self.adj[x].pop(y, None)
        self.adj.get(y, {}).pop(x, None)
        for r in (x, y):
            if r in self.adj and not self.adj[r]:
                del self.adj[r]

    def redirect(self, old: Ref, new: Ref) -> None:
        """Move all of old's bridges onto new (used when nodes are merged)."""
        for nb, s in list(self.adj.get(old, {}).items()):
            self._drop(old, nb)
            if nb != new:
                self.add(new, nb, s)

    def n_edges(self) -> int:
        return sum(len(v) for v in self.adj.values()) // 2

    def check_invariants(self) -> None:
        for a, nb in self.adj.items():
            assert len(nb) <= self.max_degree, "degree cap exceeded"
            for b, s in nb.items():
                assert self.adj[b][a] == s, "bridge not symmetric"


class MemoryStore:
    def __init__(self, max_bridge_degree: int = 4):
        self.branches: Dict[str, Branch] = {}
        self.bridges = BridgeTable(max_bridge_degree)
        self.key_peers: Dict[str, set] = defaultdict(set)
        # per peer: word -> keys whose key or value contained it (pruned lazily)
        self.words: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    def index_words(self, peer: str, key: str, value) -> None:
        idx = self.words[peer]
        for w in words_of(key) | words_of(value):
            idx[w].add(key)

    def branch(self, peer: str) -> Branch:
        if peer not in self.branches:
            self.branches[peer] = Branch(peer)
        return self.branches[peer]

    def lookup(self, peer: str, key: str):
        b = self.branches.get(peer)
        return None if b is None else b.get(key)

    def check_invariants(self) -> None:
        for b in self.branches.values():
            b.check_invariants()
        self.bridges.check_invariants()


class LogGate:
    """Organizes a peer's incoming stream and routes it to that peer's branch.

    Dedup is applied at ingest: an exact-key repeat refreshes the node (and
    records provenance / supersedes the value); with embeddings, a near
    duplicate among the shallow `near_dup_levels` depths (a constant number of
    nodes) merges into the existing node. Facts seen by several peers are
    joined by a bridge instead of being copied.
    """

    def __init__(self, store: MemoryStore, near_dup_levels: int = 3,
                 sim_threshold: float = 0.95):
        self.store = store
        self.near_dup_levels = near_dup_levels
        self.sim_threshold = sim_threshold

    def ingest(self, peer: str, key: str, value: Any, emb=None,
               t: Optional[int] = None, visibility: str = "private") -> str:
        branch = self.store.branch(peer)
        t = branch.clock + 1 if t is None else t
        hit = branch.get(key)
        if hit is not None:
            node, _ = hit
            node.mentions.append(t)
            status = "refresh"
            if node.value != value:
                node.history.append(node.value)
                node.value = value
                status = "update"
                self.store.index_words(peer, key, value)
            branch.touch(key)
            strengthen(node, branch.clock)
            return status
        if emb is not None:
            e = np.asarray(emb, dtype=float)
            for d in range(min(self.near_dup_levels, len(branch.levels))):
                for cand in branch.depth_nodes(d):
                    if cand.emb is None:
                        continue
                    cos = float(e @ cand.emb / (np.linalg.norm(e) * np.linalg.norm(cand.emb) + 1e-12))
                    if cos >= self.sim_threshold:
                        cand.mentions.append(t)
                        cand.aliases.append(key)
                        branch.aliases[key] = cand.key
                        branch.touch(cand.key)
                        strengthen(cand, branch.clock)
                        return "near_dup"
        node = Node(key=key, peer=peer, value=value,
                    emb=None if emb is None else np.asarray(emb, dtype=float),
                    visibility=visibility, created=t, mentions=[t])
        branch.insert_front(node)
        node.s_clock = branch.clock
        self.store.index_words(peer, key, value)
        for other in self.store.key_peers[key] - {peer}:
            self.store.bridges.add((peer, key), (other, key))
        self.store.key_peers[key].add(peer)
        return "new"
