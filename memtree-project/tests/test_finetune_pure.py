from memtree.finetune import (encode_example, end_token_id, segments, synthetic_examples,
                              to_messages, wants_reference)
from memtree.runtime import MEM, MEM_END, MEM_REF, MEM_REF_END, MEM_TOKENS, add_memory_tokens
from toy_tokenizer import ToyTokenizer


def _tok():
    t = ToyTokenizer()
    add_memory_tokens(t)
    return t


def test_segments_mask_only_deliveries():
    target = f"{MEM_REF}k1{MEM_REF_END}{MEM}[k1] v1{MEM_END}It is v1."
    assert segments(target) == [(f"{MEM_REF}k1{MEM_REF_END}", True),
                                (f"{MEM}[k1] v1{MEM_END}", False), ("It is v1.", True)]
    assert segments("plain answer") == [("plain answer", True)]


def test_encode_masks_prompt_and_delivery_and_trains_on_ref_and_answer():
    tok = _tok()
    ex = {"system": "Fresh memory:", "user": "What is k1?",
          "assistant": f"{MEM_REF}k1{MEM_REF_END}{MEM}[k1] v1{MEM_END}v1"}
    ids, labels = encode_example(tok, ex, 10_000)
    assert len(ids) == len(labels)
    trained = [i for i, l in zip(ids, labels) if l != -100]
    assert tok.decode(trained) == f"{MEM_REF}k1{MEM_REF_END}v1<|end|>"
    prompt = tok.apply_chat_template(to_messages(ex)[0], add_generation_prompt=True)
    assert ids[:len(prompt)] == prompt and all(l == -100 for l in labels[:len(prompt)])
    first = next(i for i, l in enumerate(labels) if l != -100)
    assert ids[first] == tok.convert_tokens_to_ids(MEM_REF)  # the decision token is trained
    assert labels[-1] == end_token_id(tok) == tok.convert_tokens_to_ids("<|end|>")


def test_encode_messages_format_and_length_limit():
    tok = _tok()
    ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    ids, labels = encode_example(tok, ex, 10_000)
    assert tok.decode([i for i, l in zip(ids, labels) if l != -100]) == "hello<|end|>"
    assert encode_example(tok, ex, 5) is None


def test_synthetic_examples_mix_both_decisions():
    ex = synthetic_examples(400, seed=1)
    assert len(ex) == 400
    n_ref = sum(map(wants_reference, ex))
    assert 0 < n_ref < 400
    tok = _tok()
    for e in ex[:50]:
        assert encode_example(tok, e, 10_000) is not None
