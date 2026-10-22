# neural-brain

A neural network that starts as random noise and learns to talk.

**No pretrained models. No downloads of weights. No local LLMs.** This is a
decoder-only transformer (`Brain`) written from scratch in ~200 lines of PyTorch,
initialized with random weights, and trained from step 0 on a single 2-core CPU
with ~2 GB of RAM.

## What is in here

```
brain/
  model.py    # the brain: 4-layer causal transformer, 1.83M parameters
  data.py     # character-level dataset (Tiny Shakespeare, 1.1M chars, vocab 65)
  train.py    # training loop: AdamW, cosine LR, grad clip, checkpointing, sampling
  sample.py   # load a checkpoint and let the brain speak
runs/brain-01/
  ckpt.pt        # latest weights
  train_log.csv  # loss / lr / throughput history
  samples.txt    # what the brain said at each checkpoint
```

## The architecture

| | |
|---|---|
| type | decoder-only transformer (GPT-style), trained for next-character prediction |
| params | 1.83M (weight-tied embedding/output head) |
| depth / width | 4 blocks / 192 dims / 6 attention heads |
| context | 192 characters |
| vocab | 65 characters |
| attention | causal, `F.scaled_dot_product_attention` |
| optimizer | AdamW (lr 6e-4, cosine decay, 300-step warmup, wd 0.1, grad clip 1.0) |
| batch | 32 sequences x 192 tokens = 6,144 tokens/step |

## Reproduce

```bash
pip install -r requirements.txt
python -m brain.train --steps 8000            # trains from scratch (~1.5h on 2 CPU cores)
python -m brain.sample --ckpt runs/brain-01/ckpt.pt --prompt "ROMEO:" --tokens 400
```

Training is fully resumable: `python -m brain.train --resume`.

## Training log

See [`runs/brain-01/train_log.csv`](runs/brain-01/train_log.csv) for the full
curve and [`runs/brain-01/samples.txt`](runs/brain-01/samples.txt) for the
brain's output at each checkpoint — from random bytes to something that reads
like a play.

_(live progress is appended below as training runs)_

## Why character-level

A character-level model has to discover words, spaces, punctuation, line breaks,
speaker labels and verse structure on its own. Watching `xqzjK` turn into
`ROMEO: I will not stay to hear thee speak.` is the whole point.
