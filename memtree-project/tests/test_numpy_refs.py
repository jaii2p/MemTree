import numpy as np

from memtree.geometry import levels_needed
from memtree.ref_numpy import (sparse_attention_dense, sparse_attention_gather,
                               sparse_indices, tree_mixer_naive,
                               tree_mixer_recurrent)


def _qkv(T, d, seed=0):
    r = np.random.default_rng(seed)
    return [r.standard_normal((T, d)) for _ in range(3)]


def test_sparse_gather_equals_dense_mask():
    for T, w, g in [(50, 8, 3), (5, 8, 3), (40, 1, 0), (33, 4, 40), (64, 16, 4)]:
        q, k, v = _qkv(T, 6, seed=T)
        out, _ = sparse_attention_gather(q, k, v, w, g)
        assert np.allclose(out, sparse_attention_dense(q, k, v, w, g), atol=1e-10)


def test_sparse_indices_valid_unique_and_linear():
    T, w, g = 200, 16, 4
    idx = sparse_indices(T, w, g)
    for t in range(T):
        row = idx[t][idx[t] >= 0]
        assert len(row) == len(set(row)) <= w + g
        assert t in row and row.max() == t
    total = lambda n: int((sparse_indices(n, w, g) >= 0).sum())
    assert total(2 * T) / total(T) < 2.2  # linear, not quadratic


def test_tree_mixer_recurrent_equals_naive():
    r = np.random.default_rng(1)
    for T, J in [(1, 1), (2, 1), (7, 3), (16, 4), (37, 6), (64, 6), (100, 7)]:
        q, k, v = _qkv(T, 5, seed=T)
        lam = r.uniform(0.5, 2.0, J + 1)
        gam = r.uniform(0.5, 1.0, max(J - 1, 0))
        a, _ = tree_mixer_recurrent(q, k, v, lam, gam)
        b = tree_mixer_naive(q, k, v, lam, gam)
        assert np.allclose(a, b, atol=1e-9), (T, J, np.abs(a - b).max())


def test_tree_mixer_is_logarithmic():
    T, J = 1024, 10
    q, k, v = _qkv(T, 3)
    lam, gam = np.ones(J + 1), np.full(J - 1, 0.95)
    _, st = tree_mixer_recurrent(q, k, v, lam, gam)
    for t in range(T):
        assert st["reads"][t] == bin(t).count("1") + 1  # popcount blocks + self
    assert st["reads"].max() <= levels_needed(T) + 1
    assert st["max_states"] <= levels_needed(T)
    assert st["merges"] < T  # amortized under one merge per token
    _, st2 = tree_mixer_recurrent(*_qkv(2 * T, 3), np.ones(J + 2), np.full(J, 0.95))
    assert st2["reads"].sum() / st["reads"].sum() < 2.3  # T log T; dense would be ~4x


def test_tree_mixer_output_is_convex_combination():
    T, J = 64, 6
    q, k, v = _qkv(T, 4)
    out, _ = tree_mixer_recurrent(q, k, v, np.ones(J + 1), np.full(J - 1, 0.9))
    for t in range(T):
        assert (out[t] <= v[: t + 1].max(0) + 1e-3).all()
        assert (out[t] >= v[: t + 1].min(0) - 1e-3).all()


def test_xor_levels_match_fenwick_matrix_including_offset_rows():
    from memtree.geometry import fenwick_level_matrix
    from memtree.ref_numpy import xor_levels
    for Tk in [1, 2, 9, 64, 300]:
        full = fenwick_level_matrix(Tk)
        for Tq in {1, min(3, Tk), Tk}:
            assert (xor_levels(Tq, Tk) == full[Tk - Tq:]).all()


def _hyb_inputs(T, d, seed=0):
    r = np.random.default_rng(seed)
    J = max(1, (T - 1).bit_length())
    return dict(q_rope=r.standard_normal((T, d)), k_rope=r.standard_normal((T, d)),
                v=r.standard_normal((T, d)), q_raw=r.standard_normal((T, d)),
                k_raw=r.standard_normal((T, d)), lam=r.uniform(0.5, 2, J + 1),
                gam=r.uniform(0.5, 1, max(J - 1, 0)), fmap=np.eye(d) + 0.1 * r.standard_normal((d, d)))


def test_hybrid_with_zero_gate_and_short_input_is_exact_full_attention():
    from memtree.ref_numpy import hybrid_rows
    x = _hyb_inputs(20, 8)
    out = hybrid_rows(x["q_rope"], x["k_rope"], x["v"], x["q_raw"], x["k_raw"], w=32, g=4,
                      lam=x["lam"], gam=x["gam"], fmap=x["fmap"], gate=0.0)
    T, d = 20, 8
    sc = x["q_rope"] @ x["k_rope"].T / np.sqrt(d)
    sc = np.where(np.tril(np.ones((T, T), bool)), sc, -np.inf)
    e = np.exp(sc - sc.max(-1, keepdims=True))
    ref = (e / e.sum(-1, keepdims=True)) @ x["v"]
    assert np.allclose(out, ref, atol=1e-12)  # T <= window: the original attention, exactly


def test_hybrid_offset_rows_equal_full_rows():
    """What the KV-cache decode path computes must match the full pass."""
    from memtree.ref_numpy import hybrid_rows
    T, d = 45, 6
    x = _hyb_inputs(T, d, seed=2)
    kw = dict(w=8, g=2, lam=x["lam"], gam=x["gam"], fmap=x["fmap"], gate=0.7)
    full = hybrid_rows(x["q_rope"], x["k_rope"], x["v"], x["q_raw"], x["k_raw"], **kw)
    for Tq in (1, 4):
        part = hybrid_rows(x["q_rope"][-Tq:], x["k_rope"], x["v"], x["q_raw"][-Tq:], x["k_raw"], **kw)
        assert np.allclose(part, full[-Tq:], atol=1e-12)


def test_hybrid_tree_branch_equals_recurrent_mixer():
    from memtree.ref_numpy import hybrid_rows, phi, tree_mixer_recurrent
    T, d = 33, 5
    x = _hyb_inputs(T, d, seed=4)
    kw = dict(lam=x["lam"], gam=x["gam"], fmap=x["fmap"])
    g1 = hybrid_rows(x["q_rope"], x["k_rope"], x["v"], x["q_raw"], x["k_raw"], w=4, g=1, gate=1.0, **kw)
    g0 = hybrid_rows(x["q_rope"], x["k_rope"], x["v"], x["q_raw"], x["k_raw"], w=4, g=1, gate=0.0, **kw)
    tree = (g1 - g0) / np.tanh(1.0)
    rec, _ = tree_mixer_recurrent(x["q_raw"] @ x["fmap"], x["k_raw"] @ x["fmap"], x["v"], x["lam"], x["gam"])
    assert np.allclose(tree, rec, atol=1e-9)
