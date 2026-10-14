import numpy as np

from memtree.geometry import (depth_of_rank, fenwick_blocks,
                              fenwick_level_matrix, levels_needed)


def test_depth_of_rank_is_geometric():
    depths = [depth_of_rank(r) for r in range(15)]
    assert depths == [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    for d in range(10):  # depth d holds exactly 2**d ranks
        assert sum(depth_of_rank(r) == d for r in range(1 << 11)) == 1 << d


def test_levels_needed():
    assert [levels_needed(n) for n in range(9)] == [0, 1, 2, 2, 3, 3, 3, 3, 4]


def test_fenwick_partition_covers_exactly():
    for n in range(0, 300):
        blocks = fenwick_blocks(n)
        assert len(blocks) == bin(n).count("1")
        pos = 0
        sizes = []
        for start, lvl in blocks:
            assert start == pos
            pos += 1 << lvl
            sizes.append(lvl)
        assert pos == n
        assert sizes == sorted(sizes, reverse=True)
        assert len(blocks) <= levels_needed(n)


def test_level_matrix():
    T = 70
    lev = fenwick_level_matrix(T)
    for t in range(T):
        assert lev[t, t] == 0
        assert (lev[t, t + 1:] == -1).all()
        assert (lev[t, :t] >= 1).all()
        assert lev[t, :t].max(initial=0) <= levels_needed(t)
