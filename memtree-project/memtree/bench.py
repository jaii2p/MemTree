"""Synthetic multi-peer long-horizon benchmark plus the cost-scaling table.

Run:  python -m memtree.bench
"""
import math
import random
from collections import Counter

import numpy as np

from .geometry import levels_needed
from .ref_numpy import sparse_indices, tree_mixer_recurrent
from .store import LogGate, MemoryStore
from .watchdog import Watchdog


# ------------------------------------------------------------------ sessions
def make_session(seed=0, n_peers=3, n_events=2000, key_space=400,
                 p_repeat=0.25, p_contra=0.3, p_query=0.25, shared_frac=0.1):
    """Events: ("write", peer, key, value) and ("query", peer, key, expected).
    Repeats restate the same value (duplicates) or change it (contradictions,
    latest wins). Some keys are shared across peers."""
    rng = random.Random(seed)
    peers = [f"peer{i}" for i in range(n_peers)]
    truth = {}
    events = []
    counter = 0
    for _ in range(n_events):
        if truth and rng.random() < p_query:
            peer, key = rng.choice(list(truth))
            events.append(("query", peer, key, truth[(peer, key)]))
            continue
        peer = rng.choice(peers)
        mine = [k for (p, k) in truth if p == peer]
        if mine and rng.random() < p_repeat:
            key = rng.choice(mine)
            if rng.random() < p_contra:
                counter += 1
                value = f"v{counter}"
            else:
                value = truth[(peer, key)]
        else:
            i = rng.randrange(key_space)
            key = f"s/k{i % 20}" if rng.random() < shared_frac else f"{peer}/k{i}"
            counter += 1
            value = truth.get((peer, key), f"v{counter}")
        truth[(peer, key)] = value
        events.append(("write", peer, key, value))
    return events


def run_session(events, boundary=2, use_watchdog=True, sweep_every=50):
    """Replay events. The main model never grabs: every reference goes to the
    watchdog, which confirms center memories and delivers outer ones.
    Verifies recall against the oracle."""
    store = MemoryStore()
    gate = LogGate(store)
    dog = Watchdog(store, boundary=boundary)
    status = Counter()
    n_q = n_ok = bound_violations = 0
    steps = []
    zone = Counter()
    for i, ev in enumerate(events):
        if ev[0] == "write":
            status[gate.ingest(ev[1], ev[2], ev[3])] += 1
        else:
            _, peer, key, expected = ev
            n_q += 1
            depth_before = store.branch(peer).index.get(key)
            d = dog.reference(peer, key, expand=True)  # every reference goes to the watchdog
            n_steps = d.hops
            if d.resolution == "in_context":
                zone["main"] += 1
                ok = store.lookup(peer, key)[0].value == expected
            else:
                zone["watchdog"] += 1
                ok = d.found and d.value == expected
            assert depth_before is None or n_steps == depth_before + 1
            n_ok += ok
            steps.append(n_steps)
            if n_steps > max(1, levels_needed(len(store.branch(peer)))):
                bound_violations += 1
        if sweep_every and i % sweep_every == 0:
            for peer in list(store.branches):
                dog.sweep(peer, budget=8)
    store.check_invariants()
    n_nodes = sum(len(b) for b in store.branches.values())
    return {
        "events": len(events), "queries": n_q, "recall": n_ok / max(n_q, 1),
        "mean_read_steps": float(np.mean(steps)) if steps else 0.0,
        "max_read_steps": max(steps) if steps else 0,
        "log_bound_violations": bound_violations,
        "status": dict(status), "reads_by_zone": dict(zone),
        "nodes": n_nodes, "bridges": store.bridges.n_edges(),
        "max_depths": max((len(b.levels) for b in store.branches.values()), default=0),
        "watchdog_steps": dog.steps,
    }


# ----------------------------------------------------------- cost scaling
def scaling_table(sizes=(64, 256, 1024, 4096), window=32, n_global=4, d=2):
    rng = np.random.default_rng(0)
    rows = []
    for T in sizes:
        J = max(1, (T - 1).bit_length())
        q, k, v = (rng.standard_normal((T, d)) for _ in range(3))
        lam = np.ones(J + 1)
        gam = np.full(max(J - 1, 0), 0.98)
        _, st = tree_mixer_recurrent(q, k, v, lam, gam)
        dense = T * (T + 1) // 2
        sparse = int((sparse_indices(T, window, n_global) >= 0).sum())
        tree = int(st["reads"].sum())
        rows.append({"T": T, "dense_keys": dense, "sparse_keys": sparse,
                     "tree_states": tree, "tree_merges": st["merges"],
                     "max_states": st["max_states"],
                     "log2T_plus_1": math.floor(math.log2(T)) + 1})
    return rows


def main():
    print("== multi-peer session ==")
    for k, v in run_session(make_session(seed=1, n_events=4000)).items():
        print(f"  {k}: {v}")
    print("\n== cost scaling (key/state reads for a length-T sequence) ==")
    print(f"{'T':>6} {'dense':>12} {'sparse':>10} {'tree':>10} {'dense/tree':>11} {'max states':>11} {'log2T+1':>8}")
    for r in scaling_table():
        print(f"{r['T']:>6} {r['dense_keys']:>12} {r['sparse_keys']:>10} {r['tree_states']:>10} "
              f"{r['dense_keys'] / r['tree_states']:>11.1f} {r['max_states']:>11} {r['log2T_plus_1']:>8}")


if __name__ == "__main__":
    main()
