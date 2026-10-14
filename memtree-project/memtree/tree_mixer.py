"""Log-linear tree mixer: replaces quadratic softmax attention.

Each query reads at most log2(t) + 1 Fenwick block states plus its own token,
so total cost is O(T log T) and state is O(log T). Block level j holds 2**j
tokens (geometric sizes); each level has a learned read weight lam[j+1] and
merging two level-j blocks applies a learned gate gam[j].

Mirrors memtree.ref_numpy.tree_mixer_* exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import fenwick_level_matrix


def phi(x):
    return F.elu(x) + 1.0


class LogLinearTreeMixer(nn.Module):
    """q, k, v: (B, H, T, d). Requires T <= 2**max_levels."""

    def __init__(self, n_heads: int, max_levels: int, eps: float = 1e-6):
        super().__init__()
        self.H, self.J, self.eps = n_heads, max_levels, eps
        self.lam_raw = nn.Parameter(torch.zeros(n_heads, max_levels + 1))
        # sigmoid(4) ~ 0.98: merges start near-lossless
        self.gam_raw = nn.Parameter(torch.full((n_heads, max(max_levels - 1, 0)), 4.0))

    def lam(self):
        return F.softplus(self.lam_raw)  # (H, J+1)

    def gam(self):
        return torch.sigmoid(self.gam_raw)  # (H, J-1)

    def _level_weights(self):
        lam, gam = self.lam(), self.gam()
        ones = torch.ones(self.H, 1, device=lam.device, dtype=lam.dtype)
        G = torch.cumprod(torch.cat([ones, gam], dim=-1), dim=-1)  # (H, J)
        return lam * torch.cat([ones, G], dim=-1)  # (H, J+1)

    def forward_dense(self, q, k, v):
        """O(T^2) explicit-weight form: for tests and gradient checks only."""
        B, H, T, _ = q.shape
        assert T <= (1 << self.J), "T must be <= 2**max_levels"
        lev = torch.as_tensor(fenwick_level_matrix(T), device=q.device)  # (T, T)
        valid = (lev >= 0).to(q.dtype)
        W = self._level_weights()[:, lev.clamp(min=0)] * valid  # (H, T, T)
        scores = torch.einsum("bhtd,bhsd->bhts", phi(q), phi(k)) * W.unsqueeze(0)
        den = scores.sum(-1, keepdim=True) + self.eps
        return torch.einsum("bhts,bhsd->bhtd", scores, v) / den

    def forward(self, q, k, v):
        """O(T log T) binary-counter form."""
        B, H, T, dk = q.shape
        dv = v.shape[-1]
        assert T <= (1 << self.J), "T must be <= 2**max_levels"
        lam, gam = self.lam(), self.gam()
        fq, fk = phi(q), phi(k)
        blocks = []  # (level, S (B,H,dk,dv), z (B,H,dk)); oldest/largest first
        outs = []
        for t in range(T):
            fqt, fkt, vt = fq[:, :, t], fk[:, :, t], v[:, :, t]
            num = torch.zeros(B, H, dv, device=q.device, dtype=q.dtype)
            den = torch.zeros(B, H, 1, device=q.device, dtype=q.dtype)
            for lvl, S, z in blocks:
                w = lam[:, lvl + 1].view(1, H, 1)
                num = num + w * torch.einsum("bhd,bhde->bhe", fqt, S)
                den = den + w * torch.einsum("bhd,bhd->bh", fqt, z).unsqueeze(-1)
            sc = (fqt * fkt).sum(-1, keepdim=True)  # (B, H, 1)
            w0 = lam[:, 0].view(1, H, 1)
            num = num + w0 * sc * vt
            den = den + w0 * sc
            outs.append(num / (den + self.eps))
            if t == T - 1:
                break
            S = torch.einsum("bhd,bhe->bhde", fkt, vt)
            z = fkt
            lvl = 0
            while blocks and blocks[-1][0] == lvl:
                _, S2, z2 = blocks.pop()
                g = gam[:, lvl].view(1, H, 1)
                S = g.unsqueeze(-1) * (S + S2)
                z = g * (z + z2)
                lvl += 1
            blocks.append((lvl, S, z))
        return torch.stack(outs, dim=2)
