"""Export model-only weights (small file) for the repo."""
import argparse
import torch
from brain.model import Brain, BrainConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/brain-01/ckpt.pt")
p.add_argument("--out", default="runs/brain-01/weights.pt")
a = p.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck["step"],
            "val_loss": ck.get("val_loss")}, a.out)
print(f"wrote {a.out} (step {ck['step']}, val {ck.get('val_loss')})")
