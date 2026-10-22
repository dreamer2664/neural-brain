"""Data: character-level stream, split into train/val."""
import os
import urllib.request

import numpy as np
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"downloading dataset -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


class CharData:
    """The brain reads and writes one character at a time."""

    def __init__(self, path, val_fraction=0.05):
        text = open(path, encoding="utf-8").read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        ids = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n = int(len(ids) * (1 - val_fraction))
        self.train = ids[:n]
        self.val = ids[n:]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]


def get_batch(ids, block_size, batch_size, seed=None):
    """Sample a batch of (input, next-char-target) pairs."""
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    # generator=None -> global RNG, so consecutive training batches are different
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,), generator=g)
    x = torch.stack([torch.from_numpy(ids[i: i + block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i + 1: i + 1 + block_size]) for i in ix])
    return x.long(), y.long()
