"""Phi-4 Mini (Phi3ForCausalLM) attention swap.

Each converted layer keeps Phi's own qkv_proj / o_proj and RoPE and replaces
the quadratic softmax over all past tokens with:

    sparse branch  softmax over a sliding window + a few global tokens (linear)
    tree branch    log-linear tree mixer over Fenwick blocks (log states)
    out = sparse + tanh(gate) * tree           gate starts at 0

At init the tree branch is switched off, so for inputs no longer than the
window the converted model computes exactly what Phi computed. Distillation
then opens the gate and trains the tree to carry what the window drops.

Math mirrors memtree.ref_numpy.hybrid_rows (tested there).

Caveats, on purpose for this phase:
  * Training-time compute uses masked dense forms (exact same math, O(T^2)
    FLOPs, memory ~ H*T^2). Decoding with the KV cache reads only the window
    for the sparse branch. Fast kernels (FlexAttention / flash window_size /
    a chunked Fenwick scan) come after the math is proven.
  * Padding masks are ignored: feed unpadded or packed sequences.
  * Works with DynamicCache (default). Static caches are not supported, since
    the cache stores RoPE'd and raw keys side by side.
"""
import contextlib
import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_mixer import LogLinearTreeMixer, phi


# ------------------------------------------------------------------ pure math
def sparse_mask(Tq: int, Tk: int, window: int, n_global: int, device) -> torch.Tensor:
    """(Tq, Tk) bool, True = attend. Query rows are the last Tq positions."""
    t = torch.arange(Tk - Tq, Tk, device=device)[:, None]
    s = torch.arange(Tk, device=device)[None, :]
    return (s <= t) & ((s > t - window) | (s < n_global))


def compute_dtype(t: torch.Tensor) -> torch.dtype:
    """At least float32; float64 stays float64."""
    return torch.promote_types(t.dtype, torch.float32)


def sparse_attention_rows(q, k, v, window: int, n_global: int):
    """q: (B,H,Tq,d), k/v: (B,H,Tk,d). Returns compute_dtype(q)."""
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    cd = compute_dtype(q)
    if Tq <= 4:  # decode: gather only the allowed keys, O(window + globals) per step
        pos = torch.arange(Tk - Tq, Tk, device=q.device)
        win = pos[:, None] - torch.arange(window - 1, -1, -1, device=q.device)[None, :]
        lo = (pos - window + 1).clamp(min=0)
        glob = torch.arange(n_global, device=q.device)[None, :].expand(Tq, n_global)
        idx = torch.cat([glob, win], dim=1)
        valid = torch.cat([glob < lo[:, None], win >= 0], dim=1)
        idx = idx.clamp(0, Tk - 1)  # invalid slots are masked; keep indices in range
        kg, vg = k[:, :, idx], v[:, :, idx]  # (B,H,Tq,W,d)
        sc = torch.einsum("bhtd,bhtwd->bhtw", q.to(cd), kg.to(cd)) / math.sqrt(d)
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        return torch.einsum("bhtw,bhtwd->bhtd", p, vg.to(cd))
    mask = sparse_mask(Tq, Tk, window, n_global, q.device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(d))
    return out.to(cd)


def xor_levels(Tq: int, Tk: int, device) -> torch.Tensor:
    """Fenwick level matrix rows for the last Tq of Tk positions:
    0 = self, j+1 = block of level j, -1 = future. float64 log2 keeps the
    highest-bit computation exact far beyond any context length."""
    t = torch.arange(Tk - Tq, Tk, device=device)[:, None]
    s = torch.arange(Tk, device=device)[None, :]
    x = torch.bitwise_xor(t, s)
    hb = torch.floor(torch.log2(x.clamp(min=1).double())).long()
    return torch.where(s < t, hb + 1, torch.where(s == t, torch.zeros_like(hb), -torch.ones_like(hb)))


def tree_rows(mixer: LogLinearTreeMixer, fq, fk, v, eps: float = 1e-6):
    """fq: (B,H,Tq,d) already feature-mapped (positive), fk: (B,H,Tk,d)."""
    Tq, Tk = fq.shape[2], fk.shape[2]
    assert Tk <= (1 << mixer.J), f"sequence {Tk} exceeds 2**max_levels"
    lev = xor_levels(Tq, Tk, fq.device)
    W = mixer._level_weights()[:, lev.clamp(min=0)] * (lev >= 0)  # (H,Tq,Tk)
    a = torch.einsum("bhtd,bhsd->bhts", fq, fk) * W.unsqueeze(0)
    return torch.einsum("bhts,bhsd->bhtd", a, v.to(a.dtype)) / (a.sum(-1, keepdim=True) + eps)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, Hkv, T, d = x.shape
    return x[:, :, None].expand(B, Hkv, n_rep, T, d).reshape(B, Hkv * n_rep, T, d)


# ------------------------------------------------------------------- module
class TreeSparseAttention(nn.Module):
    """Drop-in replacement for transformers' Phi3Attention."""

    def __init__(self, orig: nn.Module, window: int = 512, n_global: int = 4,
                 max_levels: int = 17, gate_init: float = 0.0, use_tree: bool = True):
        super().__init__()
        cfg = orig.config
        self.config = cfg
        self.layer_idx = orig.layer_idx
        self.head_dim = orig.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv
        self.window, self.n_global = window, n_global
        self.qkv_proj = orig.qkv_proj  # shared with the teacher: same weights
        self.o_proj = orig.o_proj
        self.use_tree = use_tree
        dev = orig.qkv_proj.weight.device
        pd = compute_dtype(orig.qkv_proj.weight)  # fp32 for bf16 models, fp64 for fp64
        if use_tree:
            self.mixer = LogLinearTreeMixer(self.n_heads, max_levels).to(device=dev, dtype=pd)
            self.fmap = nn.Parameter(torch.eye(self.head_dim, device=dev, dtype=pd)
                                     .repeat(self.n_heads, 1, 1))
            self.gate = nn.Parameter(torch.full((self.n_heads,), float(gate_init), device=dev, dtype=pd))
        self._teacher = [orig]  # a list, so it is not registered twice in state_dict
        self.use_teacher = False

    def extra_repr(self):
        tree = f"max_levels={self.mixer.J}" if self.use_tree else "no tree"
        return f"window={self.window}, n_global={self.n_global}, {tree}"

    def new_parameters(self):
        if not self.use_tree:
            return []
        return [self.mixer.lam_raw, self.mixer.gam_raw, self.fmap, self.gate]

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        if self.use_teacher:
            return self._teacher[0](hidden_states, position_embeddings=position_embeddings,
                                    attention_mask=attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

        B, T, _ = hidden_states.shape
        H, Hkv, hd = self.n_heads, self.n_kv, self.head_dim
        qkv = self.qkv_proj(hidden_states)
        qp, kp = H * hd, Hkv * hd
        q = qkv[..., :qp].view(B, T, H, hd).transpose(1, 2)
        k = qkv[..., qp:qp + kp].view(B, T, Hkv, hd).transpose(1, 2)
        v = qkv[..., qp + kp:].view(B, T, Hkv, hd).transpose(1, 2)
        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        if not self.use_tree:  # pure linear context: window + global tokens, Phi's own weights
            if past_key_values is not None:
                k_rope, v = past_key_values.update(k_rope, v, self.layer_idx)
            k_rope, v = repeat_kv(k_rope, self.n_rep), repeat_kv(v, self.n_rep)
            out = sparse_attention_rows(q_rope, k_rope, v, self.window, self.n_global)
            out = out.to(hidden_states.dtype).transpose(1, 2).reshape(B, T, H * hd)
            return self.o_proj(out), None

        if past_key_values is not None:
            packed = torch.cat([k_rope, k], dim=-1)  # RoPE'd keys | raw keys
            packed, v = past_key_values.update(packed, v, self.layer_idx)
            k_rope, k = packed.split(hd, dim=-1)

        k_rope, k, v = (repeat_kv(x, self.n_rep) for x in (k_rope, k, v))
        sparse = sparse_attention_rows(q_rope, k_rope, v, self.window, self.n_global)

        cd = self.gate.dtype
        g = torch.tanh(self.gate).view(1, H, 1, 1)
        # With the gate at exactly 0 the tree adds nothing, but while training the
        # gate still needs d(out)/d(gate) = tree, so it is only skipped at inference.
        training_gate = torch.is_grad_enabled() and self.gate.requires_grad
        if training_gate or bool(torch.any(g != 0)):
            fq = phi(torch.einsum("bhtd,hde->bhte", q.to(cd), self.fmap))
            fk = phi(torch.einsum("bhtd,hde->bhte", k.to(cd), self.fmap))
            out = sparse.to(cd) + g * tree_rows(self.mixer, fq, fk, v)
        else:
            out = sparse.to(cd)
        out = out.to(hidden_states.dtype)
        out = out.transpose(1, 2).reshape(B, T, H * hd)
        return self.o_proj(out), None


# --------------------------------------------------------------- conversion
def convert_phi3(model, window: int = 512, n_global: int = 4, max_levels: int = 17,
                 keep_full_layers: Sequence[int] = (), gate_init: float = 0.0,
                 use_tree: bool = True):
    """Swap attention in place. keep_full_layers stay original softmax.
    use_tree=False: linear context only (window + global tokens) on Phi's own
    weights, with no compressed summaries; older context belongs to the watchdog."""
    layers = model.model.layers
    for i, layer in enumerate(layers):
        if i in keep_full_layers or isinstance(layer.self_attn, TreeSparseAttention):
            continue
        layer.self_attn = TreeSparseAttention(layer.self_attn, window, n_global,
                                              max_levels, gate_init, use_tree)
    model.config.memtree = {"window": window, "n_global": n_global, "max_levels": max_levels,
                            "keep_full_layers": list(keep_full_layers), "use_tree": use_tree}
    return model


def converted_layers(model) -> Iterable[TreeSparseAttention]:
    for m in model.modules():
        if isinstance(m, TreeSparseAttention):
            yield m


def new_parameters(model):
    return [p for m in converted_layers(model) for p in m.new_parameters()]


@contextlib.contextmanager
def teacher_mode(model, on: bool = True):
    """Run converted layers as the original Phi attention (for targets)."""
    mods = list(converted_layers(model))
    prev = [m.use_teacher for m in mods]
    for m in mods:
        m.use_teacher = on
    try:
        yield model
    finally:
        for m, p in zip(mods, prev):
            m.use_teacher = p


ADAPTER_SUFFIXES = (".self_attn.mixer.lam_raw", ".self_attn.mixer.gam_raw",
                    ".self_attn.fmap", ".self_attn.gate")


def save_adapter(model, path: str):
    ids = {id(p) for p in new_parameters(model)}
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if id(p) in ids}
    assert all(n.endswith(ADAPTER_SUFFIXES) for n in state), "unexpected adapter param"
    torch.save({"config": model.config.memtree, "state": state}, path)


def load_adapter(model, path: str):
    ck = torch.load(path, map_location="cpu")
    convert_phi3(model, **ck["config"])
    missing = set(ck["state"])
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in ck["state"].items():
            assert n.endswith(ADAPTER_SUFFIXES), n
            params[n].copy_(t.to(params[n].device, params[n].dtype))
            missing.discard(n)
    assert not missing
    return model


def window_model(model, window: int = 1024, n_global: int = 4):
    """Phi with linear context only: no tree, nothing to train, no adapter file.
    Exact long-range memory is the watchdog's job (memtree.runtime)."""
    return convert_phi3(model, window=window, n_global=n_global, use_tree=False)
