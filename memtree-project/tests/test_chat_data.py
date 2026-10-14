from memtree.chat_data import CANNED_FILLER, build_dataset
from memtree.finetune import encode_example, segments, wants_reference
from memtree.runtime import MEM, MEM_END, MEM_REF, MEM_REF_END, add_memory_tokens
from toy_tokenizer import ToyTokenizer


def test_dataset_mixes_behaviours_and_deliveries_are_correct():
    ex, st = build_dataset(200, seed=2)
    assert st["reference"] > 30 and st["direct"] > 100 and st["filler"] > 100 and st["never_told"] > 5
    assert st["dropped"] < 0.1 * (st["reference"] + st["direct"])
    for e in ex:
        a = e["assistant"]
        assert a.strip() and not a.strip().startswith("v")
        if a.startswith(MEM_REF):
            words = a[len(MEM_REF):a.index(MEM_REF_END)]
            assert words.strip() and not words.startswith("turn")  # plain words, never keys
            assert MEM in a and MEM_END in a
            tail = a[a.index(MEM_END) + len(MEM_END):]
            assert len(tail) > 10  # a real sentence after the delivery
        assert e["system"].startswith("Fresh memory:")


def test_examples_encode_with_delivery_masked():
    tok = ToyTokenizer()
    add_memory_tokens(tok)
    ex, _ = build_dataset(30, seed=3)
    ref = next(e for e in ex if wants_reference(e))
    ids, labels = encode_example(tok, ref, 10 ** 9)
    trained = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    assert MEM not in trained and trained.startswith(MEM_REF)
    assert all(encode_example(tok, e, 10 ** 9) for e in ex)


def test_passage_dialogues_trust_visible_answers_and_reference_the_rest():
    from memtree.chat_data import CANNED_FILLER, build_passage_dialogue
    import random
    passages = [" ".join(f"Line {i} of an old sea story about ropes, tides and gulls." for i in range(400))]
    rng = random.Random(0)
    tot = {}
    for _ in range(150):
        ex, st = build_passage_dialogue(rng, CANNED_FILLER, passages, words=(300, 1500))
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        for e in ex:
            if e["assistant"].startswith(MEM_REF):
                assert MEM in e["assistant"]
    assert tot["direct"] + tot["reference"] > 0.7 * tot["passage"]
