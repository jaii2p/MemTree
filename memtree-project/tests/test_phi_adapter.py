"""Tiny random Phi3 model: needs torch + transformers, runs in seconds on CPU."""
import unittest

import numpy as np


def _libs():
    try:
        import torch
        import transformers  # noqa: F401
        return torch
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")


def _tiny(torch, layers=2):
    from transformers import Phi3Config, Phi3ForCausalLM
    torch.manual_seed(0)
    cfg = Phi3Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=1024, partial_rotary_factor=0.75,
                     rope_scaling=None, tie_word_embeddings=True, pad_token_id=0)
    return Phi3ForCausalLM(cfg).double().eval()


def test_math_matches_numpy_reference():
    torch = _libs()
    from memtree.phi_adapter import sparse_attention_rows, tree_rows, xor_levels
    from memtree.ref_numpy import hybrid_rows, xor_levels as np_xor
    from memtree.tree_mixer import LogLinearTreeMixer, phi
    assert (xor_levels(5, 77, "cpu").numpy() == np_xor(5, 77)).all()
    T, d, H = 30, 8, 2
    g = torch.Generator().manual_seed(0)
    qr, kr, v, qw, kw = (torch.randn(1, H, T, d, generator=g, dtype=torch.float64) for _ in range(5))
    mixer = LogLinearTreeMixer(H, 5).double()
    with torch.no_grad():
        mixer.lam_raw.normal_()
    fmap = torch.eye(d, dtype=torch.float64).repeat(H, 1, 1)
    for Tq in (T, 1, 3):  # prefill (SDPA) path and decode (gather) path
        sp = sparse_attention_rows(qr[:, :, -Tq:], kr, v, 6, 2)
        tr = tree_rows(mixer, phi(qw[:, :, -Tq:]), phi(kw), v)
        out = (sp + np.tanh(0.5) * tr).detach()
        for h in range(H):
            ref = hybrid_rows(qr[0, h, -Tq:].numpy(), kr[0, h].numpy(), v[0, h].numpy(),
                              qw[0, h, -Tq:].numpy(), kw[0, h].numpy(), 6, 2,
                              mixer.lam()[h].detach().numpy(), mixer.gam()[h].detach().numpy(),
                              fmap[h].numpy(), 0.5)
            assert np.allclose(out[0, h].numpy(), ref, atol=1e-9), (Tq, h)


def test_converted_equals_original_at_init_for_short_inputs():
    torch = _libs()
    from memtree.phi_adapter import convert_phi3, teacher_mode
    model = _tiny(torch)
    ids = torch.randint(1, 97, (2, 20))
    with torch.no_grad():
        before = model(ids).logits
        convert_phi3(model, window=32, n_global=4, max_levels=10)
        after = model(ids).logits
        with teacher_mode(model):
            teacher = model(ids).logits
    assert torch.allclose(before, after, atol=1e-8)
    assert torch.allclose(before, teacher, atol=1e-12)


def test_kv_cache_decode_matches_full_forward():
    torch = _libs()
    from memtree.phi_adapter import convert_phi3, converted_layers
    model = _tiny(torch)
    convert_phi3(model, window=8, n_global=2, max_levels=10)
    with torch.no_grad():
        for m in converted_layers(model):
            m.gate.fill_(0.8)  # tree branch on
            m.mixer.lam_raw.normal_()
    ids = torch.randint(1, 97, (1, 41))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        out = model(ids[:, :-3], use_cache=True)
        pkv = out.past_key_values
        steps = [out.logits[:, -1]]
        for i in range(3, 0, -1):
            o = model(ids[:, -i:-i + 1] if i > 1 else ids[:, -1:], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[:, -1])
    for j, s in enumerate(steps):
        assert torch.allclose(s, full[:, 41 - 4 + j], atol=1e-8), j
    a = model.generate(ids[:, :10], max_new_tokens=12, do_sample=False, use_cache=True)
    b = model.generate(ids[:, :10], max_new_tokens=12, do_sample=False, use_cache=False)
    assert torch.equal(a, b)


def test_gate_gets_gradient_at_zero_and_distill_steps_reduce_loss():
    torch = _libs()
    from memtree.distill import freeze_all_but_adapter, layerwise_step
    from memtree.phi_adapter import convert_phi3, converted_layers
    model = _tiny(torch)
    convert_phi3(model, window=4, n_global=1, max_levels=10)
    params = freeze_all_but_adapter(model)
    ids = torch.randint(1, 97, (1, 64))
    out = model(ids).logits.sum()
    out.backward()
    for m in converted_layers(model):
        assert m.gate.grad is not None and m.gate.grad.abs().sum() > 0
    for p in params:
        p.grad = None
    opt = torch.optim.Adam(params, lr=3e-2)
    losses = [layerwise_step(model, ids, opt, params) for _ in range(30)]
    assert losses[-1] < losses[0]


def test_save_and_load_adapter_roundtrip(tmp_path=None):
    torch = _libs()
    import os
    import tempfile
    from memtree.phi_adapter import convert_phi3, converted_layers, load_adapter, save_adapter
    model = _tiny(torch)
    convert_phi3(model, window=8, n_global=2, max_levels=10, keep_full_layers=[1])
    assert len(list(converted_layers(model))) == 1
    with torch.no_grad():
        for m in converted_layers(model):
            m.gate.fill_(0.3)
    path = os.path.join(tempfile.mkdtemp(), "a.pt")
    save_adapter(model, path)
    fresh = _tiny(torch)
    load_adapter(fresh, path)
    ids = torch.randint(1, 97, (1, 30))
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, fresh(ids).logits, atol=1e-10)


def test_window_model_no_tree_matches_original_within_window_and_caches_correctly():
    torch = _libs()
    from memtree.phi_adapter import converted_layers, new_parameters, window_model
    model = _tiny(torch)
    short = torch.randint(1, 97, (1, 20))
    with torch.no_grad():
        before = model(short).logits
    window_model(model, window=32, n_global=2)
    assert new_parameters(model) == []  # no box: nothing new to train
    assert all(not m.use_tree for m in converted_layers(model))
    long = torch.randint(1, 97, (1, 70))
    with torch.no_grad():
        assert torch.allclose(model(short).logits, before, atol=1e-8)
        full = model(long, use_cache=False).logits
        out = model(long[:, :-1], use_cache=True)
        step = model(long[:, -1:], past_key_values=out.past_key_values, use_cache=True)
    assert torch.allclose(step.logits[:, -1], full[:, -1], atol=1e-8)
    a = model.generate(long[:, :40], max_new_tokens=10, do_sample=False, use_cache=True)
    b = model.generate(long[:, :40], max_new_tokens=10, do_sample=False, use_cache=False)
    assert torch.equal(a, b)
