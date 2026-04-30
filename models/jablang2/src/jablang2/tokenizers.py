import json

import jax.numpy as jnp
import numpy as np

from .vocab import ablang_vocab


class ABtokenizer:
    """
    Tokenizer for the heavy/light chain of antibodies.
    JAX-compatible version.
    """

    def __init__(self, vocab_dir=None):
        self.set_vocab(vocab_dir)

    def __call__(self, sequence_list, mode='encode', pad=False, w_extra_tkns=True):
        if w_extra_tkns:
            sequence_list = [sequence_list] if isinstance(sequence_list[0], str) else sequence_list
        else:
            sequence_list = [sequence_list] if isinstance(sequence_list, str) else sequence_list

        if mode == 'encode':
            data = [self.encode(seq, w_extra_tkns=w_extra_tkns) for seq in sequence_list]
            if pad:
                max_len = max(len(d) for d in data)
                data = [
                    jnp.pad(d, ((0, max_len - len(d)),), mode='constant', constant_values=self.pad_token)
                    for d in data
                ]
                return jnp.array(data, dtype=jnp.int32)
            else:
                return [jnp.array(d, dtype=jnp.int32) for d in data]
        elif mode == 'decode':
            return [self.decode(tokenized_seq) for tokenized_seq in sequence_list]
        else:
            raise SyntaxError("Given mode doesn't exist. Use either encode or decode.")

    def set_vocab(self, vocab_dir):
        if vocab_dir:
            with open(vocab_dir, encoding="utf-8") as vocab_handle:
                self.aa_to_token = json.load(vocab_handle)
        else:
            self.aa_to_token = ablang_vocab

        self.token_to_aa = {v: k for k, v in self.aa_to_token.items()}
        self.pad_token = self.aa_to_token['-']
        self.start_token = self.aa_to_token['<']
        self.end_token = self.aa_to_token['>']
        self.sep_token = self.aa_to_token['|']
        self.mask_token = self.aa_to_token['*']
        self.unknown_token = self.aa_to_token['X']
        self.all_special_tokens = [
            self.pad_token,
            self.start_token,
            self.end_token,
            self.sep_token,
            self.mask_token,
            self.unknown_token
        ]

    def encode(self, sequence, w_extra_tkns=True):
        if w_extra_tkns:
            heavy, light = sequence
            sequence = f"<{heavy}>|<{light}>".replace("<>","")

        tokenized_seq = [self.aa_to_token[resn] for resn in sequence]
        return jnp.array(tokenized_seq, dtype=jnp.int32)

    def decode(self, tokenized_seq):
        if hasattr(tokenized_seq, 'tolist'):
            tokenized_seq = tokenized_seq.tolist()

        return ''.join([self.token_to_aa[token] for token in tokenized_seq])
