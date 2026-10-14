"""Run on a machine with torch installed; skipped otherwise."""
import unittest

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise unittest.SkipTest("torch not installed")


def test_torch_sparse_attention_matches_numpy_reference():
    torch = _torch()
    from memtree.ref_numpy import sparse_attention_dense
    from memtree.sparse_attn import sparse_attention
    torch.manual_seed(0)
    B, H, T, d = 2, 3, 40, 8
    q, k, v = (torch.randn(B, H, T, d, dtype=torch.float64) for _ in range(3))
    out = sparse_attention(q, k, v, window=6, n_global=2)
    for b in range(B):
        for h in range(H):
            ref = sparse_attention_dense(q[b, h].numpy(), k[b, h].numpy(), v[b, h].numpy(), 6, 2)
            assert np.allclose(out[b, h].numpy(), ref, atol=1e-9)


def test_torch_mixer_recurrent_equals_dense_and_numpy():
    torch = _torch()
    from memtree.ref_numpy import tree_mixer_naive
    from memtree.tree_mixer import LogLinearTreeMixer
    torch.manual_seed(0)
    B, H, d = 2, 3, 6
    for T, J in [(1, 1), (9, 4), (16, 4), (37, 6)]:
        m = LogLinearTreeMixer(H, J).double()
        with torch.no_grad():
            m.lam_raw.normal_()
            m.gam_raw.normal_(mean=1.0)
        q, k, v = (torch.randn(B, H, T, d, dtype=torch.float64) for _ in range(3))
        rec, dense = m(q, k, v), m.forward_dense(q, k, v)
        assert torch.allclose(rec, dense, atol=1e-9), (T, J)
        lam, gam = m.lam().detach().numpy(), m.gam().detach().numpy()
        for b in range(B):
            for h in range(H):
                ref = tree_mixer_naive(q[b, h].numpy(), k[b, h].numpy(), v[b, h].numpy(),
                                       lam[h], gam[h])
                assert np.allclose(rec[b, h].numpy(), ref, atol=1e-9)


def test_torch_mixer_gradients_flow_to_all_parameters():
    torch = _torch()
    from memtree.tree_mixer import LogLinearTreeMixer
    m = LogLinearTreeMixer(2, 5)
    q, k, v = (torch.randn(1, 2, 20, 4, requires_grad=True) for _ in range(3))
    m(q, k, v).sum().backward()
    for p in (m.lam_raw, m.gam_raw, q, k, v):
        assert p.grad is not None and torch.isfinite(p.grad).all()
