"""Load a checkpoint and let the brain talk."""
import argparse

import torch

from brain.data import CharData, prepare
from brain.model import Brain, BrainConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/brain-01/ckpt.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = BrainConfig(**ck["cfg"])
    model = Brain(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    data = CharData(prepare())
    ids = data.encode(args.prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, args.tokens, args.temperature, args.top_k)
    print(data.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
