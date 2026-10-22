"""Sanity checks for the brain. Run: python tests.py"""
import torch

from brain.data import CharData, get_batch, prepare
from brain.model import Brain, BrainConfig

torch.manual_seed(0)
data = CharData(prepare())

# 1. batches must differ between calls (a fixed generator would repeat one batch forever)
a, _ = get_batch(data.train, 64, 4)
b, _ = get_batch(data.train, 64, 4)
assert not torch.equal(a, b), "training batches are not being resampled"

# 2. targets are inputs shifted by one character
x, y = get_batch(data.train, 64, 4, seed=1)
assert torch.equal(x[:, 1:], y[:, :-1])

# 3. loss of a fresh brain is ~ln(vocab): it knows nothing yet
cfg = BrainConfig(vocab_size=data.vocab_size, block_size=64, n_layer=2, n_head=2,
                  n_embd=64, dropout=0.0)
model = Brain(cfg).eval()
with torch.no_grad():
    _, loss = model(x, y)
import math
assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3, f"init loss {loss.item():.3f}"

# 4. causality: a token at position t must not affect logits before t
with torch.no_grad():
    base = model(x)[0]
    x2 = x.clone()
    x2[:, 40] = (x2[:, 40] + 5) % cfg.vocab_size
    pert = model(x2)[0]
assert torch.allclose(base[:, :40], pert[:, :40], atol=1e-6), "attention is not causal"
assert not torch.allclose(base[:, 40:], pert[:, 40:], atol=1e-6), "future tokens ignored"

# 5. generation appends exactly the requested number of tokens
out = model.generate(x[:1, :10], 7)
assert out.shape == (1, 17)

print("all tests passed")
