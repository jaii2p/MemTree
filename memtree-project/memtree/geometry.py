"""Shared geometry for the geometric tree and the Fenwick partition.

Geometric tree (store): items ranked by recency (rank 0 = newest) live at
depth floor(log2(rank + 1)). Depth d therefore holds exactly 2**d items (the
last occupied depth may be partial), so n items span n.bit_length() depths.

Fenwick partition (mixer): the n tokens before the current one are split into
power-of-two blocks following the set bits of n, oldest/largest block first.
"""
import numpy as np


def depth_of_rank(rank: int) -> int:
    """Depth of the item with `rank` newer items in front of it."""
    return (rank + 1).bit_length() - 1


def level_capacity(depth: int) -> int:
    return 1 << depth


def levels_needed(n: int) -> int:
    """Number of occupied depths for n items (0 for n == 0)."""
    return n.bit_length()


def fenwick_blocks(n: int):
    """Partition positions [0, n) into blocks. Returns [(start, level)], block
    size is 2**level, ordered oldest (largest) first. Block count == popcount(n).
    """
    blocks = []
    start = 0
    for j in range(n.bit_length() - 1, -1, -1):
        if (n >> j) & 1:
            blocks.append((start, j))
            start += 1 << j
    return blocks


def fenwick_level_matrix(T: int) -> np.ndarray:
    """lev[t, s]: -1 if s > t (future), 0 if s == t (the token itself), and
    j + 1 if s lies in a Fenwick block of level j in the partition of [0, t).
    """
    lev = np.full((T, T), -1, dtype=np.int64)
    for t in range(T):
        lev[t, t] = 0
        for start, j in fenwick_blocks(t):
            lev[t, start:start + (1 << j)] = j + 1
    return lev
