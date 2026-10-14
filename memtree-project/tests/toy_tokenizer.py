"""Character-level stand-in for the Phi tokenizer (special tokens are atomic)."""
import re

SPECIALS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class ToyTokenizer:
    def __init__(self):
        self.vocab = list(SPECIALS) + [chr(c) for c in range(32, 127)] + ["\n"]
        self.additional_special_tokens = []
        self.eos_token_id = 0
        self.unk_token_id = None
        self._refresh()

    def _refresh(self):
        self.ids = {t: i for i, t in enumerate(self.vocab)}
        specials = SPECIALS + self.additional_special_tokens
        self._re = re.compile("(" + "|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True)) + ")")

    def __len__(self):
        return len(self.vocab)

    def add_special_tokens(self, d, replace_additional_special_tokens=True):
        new = [t for t in d.get("additional_special_tokens", []) if t not in self.ids]
        self.vocab += new
        self.additional_special_tokens += [t for t in d["additional_special_tokens"]
                                           if t not in self.additional_special_tokens]
        self._refresh()
        return len(new)

    def convert_tokens_to_ids(self, t):
        return self.ids.get(t)

    def encode(self, text):
        out = []
        for part in self._re.split(text):
            if part in self.ids and (part in SPECIALS or part in self.additional_special_tokens):
                out.append(self.ids[part])
            else:
                out += [self.ids.get(c, self.ids["?"]) for c in part]
        return out

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        sp = set(SPECIALS + self.additional_special_tokens)
        ids = [i for i in ids if 0 <= i < len(self.vocab)]  # ids beyond the vocab: drop
        return "".join(self.vocab[i] for i in ids if not (skip_special_tokens and self.vocab[i] in sp))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True,
                            return_tensors=None):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return torch.tensor([ids])
        return ids
