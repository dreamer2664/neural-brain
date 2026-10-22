#!/usr/bin/env bash
# Train the brain from scratch, from step zero.
set -e
cd "$(dirname "$0")"
python3 -m brain.train \
  --out runs/brain-01 \
  --steps 8000 \
  --batch-size 32 \
  --block-size 192 \
  --n-layer 4 --n-head 6 --n-embd 192 \
  --lr 6e-4 \
  --eval-every 200 --ckpt-every 400 --sample-every 400 \
  "$@"
