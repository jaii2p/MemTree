"""numpy reference implementations. These define the math; the torch modules
must match them. Single sequence, single head: q, k, v are (T, d) arrays.
"""
import numpy as np

from .geometry import fenwick_level_matrix


# ---------------------------------------------------------------- sparse attention
def sparse_indices(T: int, w: int, g: int) -> np.ndarray:
    """(T, w + g) key indices per query, -1 padded. Globals (first g positions
    that fall before the window) come first, then the causal window of w."""
    idx = np.full((T, w + g), -1, dtype=np.int64)
    for t in range(T):
        lo = max(0, t - w + 1)
        row = list(range(min(g, lo))) + list(range(lo, t + 1))
        idx[t, :len(row)] = row
    return idx


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def sparse_attention_dense(q, k, v, w, g):
    """O(T^2) reference: full score matrix with an explicit sparsity mask."""
    T, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    s = np.arange(T)[None, :]
    t = np.arange(T)[:, None]
    allowed = (s <= t) & ((s > t - w) | (s < g))
    scores = np.where(allowed, scores, -np.inf)
    return _softmax(scores) @ v


def sparse_attention_gather(q, k, v, w, g):
    """O(T * (w + g)): each query only touches its own index set.
    Returns (out, visits) where visits[t] is the number of keys read."""
    T, d = q.shape
    idx = sparse_indices(T, w, g)
    out = np.zeros_like(v, dtype=np.float64)
    visits = np.zeros(T, dtype=np.int64)
    for t in range(T):
        sel = idx[t][idx[t] >= 0]
        visits[t] = len(sel)
        p = _softmax(q[t] @ k[sel].T / np.sqrt(d))
        out[t] = p @ v[sel]
    return out, visits


# ------------------------------------------------------------- log-linear tree mixer
def xor_levels(Tq: int, Tk: int) -> np.ndarray:
    """Rows Tk-Tq .. Tk-1 of fenwick_level_matrix(Tk), computed directly:
    for s < t the Fenwick block level is the highest set bit of t xor s."""
    t = np.arange(Tk - Tq, Tk)[:, None]
    s = np.arange(Tk)[None, :]
    x = np.bitwise_xor(t, s)
    hb = np.floor(np.log2(np.maximum(x, 1).astype(np.float64))).astype(np.int64)
    return np.where(s < t, hb + 1, np.where(s == t, 0, -1))


def phi(x):
    """elu(x) + 1: strictly positive feature map."""
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def _level_weights(lam, gam):
    """Per-level weight w[m] for m in 0..J. m = 0 is the token itself, m >= 1
    is a Fenwick block of level m - 1, weight lam[m] * prod_{i < m-1} gam[i]."""
    J = len(lam) - 1
    assert len(gam) == max(J - 1, 0)
    G = np.cumprod(np.concatenate([[1.0], gam]))  # G[j] = prod_{i<j} gam[i], len J
    return lam * np.concatenate([[1.0], G])  # len J + 1


def tree_mixer_naive(q, k, v, lam, gam, eps=1e-6):
    """O(T^2) reference built from the explicit level matrix."""
    T = q.shape[0]
    J = len(lam) - 1
    assert T <= (1 << J), "T must be <= 2**J"
    lev = fenwick_level_matrix(T)
    wlev = _level_weights(lam, gam)
    W = np.where(lev >= 0, wlev[np.maximum(lev, 0)], 0.0)
    scores = (phi(q) @ phi(k).T) * W
    return (scores @ v) / (scores.sum(-1, keepdims=True) + eps)


def tree_mixer_recurrent(q, k, v, lam, gam, eps=1e-6):
    """O(T log T) binary-counter version: at most log2(T)+1 states, one
    amortized merge per token. Returns (out, stats)."""
    T, dk = q.shape
    dv = v.shape[1]
    J = len(lam) - 1
    assert T <= (1 << J), "T must be <= 2**J"
    fq, fk = phi(q), phi(k)
    blocks = []  # (level, S[dk, dv], z[dk]); oldest/largest first
    out = np.zeros((T, dv))
    reads = np.zeros(T, dtype=np.int64)
    merges = 0
    max_states = 0
    for t in range(T):
        num = np.zeros(dv)
        den = 0.0
        for lvl, S, z in blocks:
            num += lam[lvl + 1] * (fq[t] @ S)
            den += lam[lvl + 1] * (fq[t] @ z)
        sc = fq[t] @ fk[t]
        num += lam[0] * sc * v[t]
        den += lam[0] * sc
        out[t] = num / (den + eps)
        reads[t] = len(blocks) + 1
        max_states = max(max_states, len(blocks))
        if t == T - 1:
            break  # no read follows, so no push needed
        S = np.outer(fk[t], v[t])
        z = fk[t].copy()
        lvl = 0
        while blocks and blocks[-1][0] == lvl:
            _, S2, z2 = blocks.pop()
            S = gam[lvl] * (S + S2)
            z = gam[lvl] * (z + z2)
            lvl += 1
            merges += 1
        blocks.append((lvl, S, z))
    return out, {"reads": reads, "merges": merges, "max_states": max_states}


# ----------------------------------------------------------- hybrid (adapter math)
def hybrid_rows(q_rope, k_rope, v, q_raw, k_raw, w, g, lam, gam, fmap, gate, eps=1e-6):
    """One head of the adapter: output rows for the last Tq of Tk positions.

    sparse branch: softmax attention on RoPE'd q/k over window + globals.
    tree branch:   log-linear mixer on phi(fmap @ raw q/k), Fenwick levels.
    out = sparse + tanh(gate) * tree
    q_*: (Tq, d), k_*/v: (Tk, d), fmap: (d, d).
    """
    Tq, d = q_rope.shape
    Tk = k_rope.shape[0]
    t = np.arange(Tk - Tq, Tk)[:, None]
    s = np.arange(Tk)[None, :]
    allowed = (s <= t) & ((s > t - w) | (s < g))
    sc = np.where(allowed, q_rope @ k_rope.T / np.sqrt(d), -np.inf)
    sparse = _softmax(sc) @ v
    lev = xor_levels(Tq, Tk)
    wlev = _level_weights(lam, gam)
    W = np.where(lev >= 0, wlev[np.maximum(lev, 0)], 0.0)
    a = (phi(q_raw @ fmap) @ phi(k_raw @ fmap).T) * W
    tree = (a @ v) / (a.sum(-1, keepdims=True) + eps)
    return sparse + np.tanh(gate) * tree
