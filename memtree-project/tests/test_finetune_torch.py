"""Tiny random Phi3 + toy tokenizer: needs torch + transformers, CPU, seconds."""
import os
import tempfile
import unittest

from toy_tokenizer import ToyTokenizer


def _libs():
    try:
        import torch
        import transformers  # noqa: F401
        return torch
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")


def _tiny(torch, vocab=101):
    from transformers import Phi3Config, Phi3ForCausalLM
    torch.manual_seed(0)
    cfg = Phi3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=2048, partial_rotary_factor=0.75,
                     rope_scaling=None, tie_word_embeddings=True, pad_token_id=0)
    return Phi3ForCausalLM(cfg).eval()


def _setup(torch, convert=True):
    from memtree.phi_adapter import convert_phi3
    model, tok = _tiny(torch), ToyTokenizer()  # 101 base tokens; memory tokens force a resize
    if convert:
        convert_phi3(model, window=16, n_global=2, max_levels=11)
    return model, tok


def test_prepare_leaves_outputs_unchanged_except_new_token_rows():
    torch = _libs()
    from memtree.finetune import prepare
    model, tok = _setup(torch)
    ids = torch.tensor([tok.encode("hello there, general memory")])
    with torch.no_grad():
        before = model(ids).logits
    params = prepare(model, tok, r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        after = model(ids).logits
    mem = model.config.memtree_finetune["token_ids"]
    assert after.shape[-1] == len(tok) == 105 and min(mem) == 101  # embeddings resized
    keep = [i for i in range(before.shape[-1]) if i not in mem]
    assert torch.allclose(before[..., keep], after[..., keep], atol=1e-6)
    assert all(p.requires_grad for p in params)
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if not n.endswith((".lora_A", ".lora_B", "d_in", "d_out")))


def test_training_teaches_reference_decision_and_roundtrips():
    torch = _libs()
    from memtree.finetune import (encode_example, evaluate, load_finetune, prepare,
                                  save_finetune, synthetic_examples, train)
    model, tok = _setup(torch)
    params = prepare(model, tok, r=8, alpha=16, dropout=0.0)
    exs = synthetic_examples(40, seed=3)
    enc = [encode_example(tok, e, 20000) for e in exs]
    assert all(e is not None for e in enc)
    before = evaluate(model, tok, enc[:8], exs[:8], "cpu")
    hist = train(model, tok, enc, [], params, steps=40, lr=3e-3, accum=2, warmup=5,
                 device="cpu", log_every=1000)
    after = evaluate(model, tok, enc[:8], exs[:8], "cpu")
    assert after["loss"] < before["loss"] and hist[-1] < hist[0]
    path = os.path.join(tempfile.mkdtemp(), "ft.pt")
    save_finetune(model, path)
    fresh, tok2 = _setup(torch)
    load_finetune(fresh, tok2, path)
    x = torch.tensor([enc[0][0]])
    with torch.no_grad():
        assert torch.allclose(model(x).logits, fresh(x).logits, atol=1e-5)


def test_runtime_runs_on_finetuned_model():
    torch = _libs()
    from memtree.finetune import prepare
    from memtree.runtime import MemoryRuntime
    model, tok = _setup(torch)
    prepare(model, tok, r=4, alpha=8, dropout=0.0)
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|end|>"), 0]
    rt = MemoryRuntime(model, tok)
    r = rt.chat("me", "remember: my dog is Biscuit", max_new_tokens=8)
    assert isinstance(r["reply"], str) and len(rt.store.branch("me")) == 2


def test_prepare_twice_is_refused():
    torch = _libs()
    from memtree.finetune import prepare
    model, tok = _setup(torch)
    prepare(model, tok, r=4, alpha=8, dropout=0.0)
    try:
        prepare(model, tok, r=4, alpha=8, dropout=0.0)
    except RuntimeError:
        return
    raise AssertionError("second prepare should raise")


def test_runtime_overflow_files_long_message_on_window_model():
    torch = _libs()
    from memtree.finetune import prepare
    from memtree.phi_adapter import window_model
    from memtree.runtime import FILED_NOTE, MemoryRuntime
    model, tok = _tiny(torch), ToyTokenizer()
    window_model(model, window=64, n_global=2)
    prepare(model, tok, r=4, alpha=8, dropout=0.0)
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|end|>"), 0]
    rt = MemoryRuntime(model, tok, window=300)
    long = " ".join(f"Line {i} is here." for i in range(120)) + " The code for Harlan is 48213. Ok?"
    r = rt.chat("me", long, max_new_tokens=6)
    keys = list(rt.store.branch("me").index)
    assert any(k.startswith("turn1:part") for k in keys) and "turn1:user" in keys
    assert isinstance(r["reply"], str)
