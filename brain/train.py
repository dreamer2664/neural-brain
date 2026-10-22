"""
Train the brain from scratch.

  python -m brain.train --steps 20000

Checkpoints every --ckpt-every steps, resumable, logs a CSV + text samples so you
can watch the thing learn to talk.
"""
import argparse
import csv
import math
import os
import time

import torch

from brain.data import CharData, get_batch, prepare
from brain.model import Brain, BrainConfig


def lr_at(step, args):
    """Warmup then cosine decay to 10% of peak."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step > args.steps:
        return args.lr * 0.1
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def estimate_loss(model, data, args):
    model.eval()
    out = {}
    for split, ids in (("train", data.train), ("val", data.val)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(ids, model.cfg.block_size, args.batch_size, seed=1234 + k)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_text(model, data, prompt="\n", n=240, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor([data.encode(prompt) or [0]], dtype=torch.long)
    out = model.generate(idx, n, temperature=temperature, top_k=top_k)
    txt = data.decode(out[0].tolist())
    model.train()
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/brain-01")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=192)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time-budget", type=float, default=0.0,
                   help="stop after N seconds (0 = no limit)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out, exist_ok=True)

    data = CharData(prepare())
    cfg = BrainConfig(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Brain(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    start_step = 0
    best_val = float("inf")
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

    log_path = os.path.join(args.out, "train_log.csv")
    new_log = not os.path.exists(log_path) or start_step == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tok_per_sec", "elapsed_s"])

    def save(step, val):
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "best_val": best_val,
                "args": vars(args),
                "val_loss": val,
            },
            ckpt_path,
        )

    print(
        f"brain: {model.num_params()/1e6:.2f}M params | vocab {cfg.vocab_size} | "
        f"ctx {cfg.block_size} | batch {args.batch_size} | tokens/step "
        f"{args.batch_size * cfg.block_size} | steps {args.steps}",
        flush=True,
    )

    t0 = time.time()
    tokens_per_step = args.batch_size * cfg.block_size
    model.train()
    for step in range(start_step, args.steps + 1):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, data, args)
            elapsed = time.time() - t0
            tps = tokens_per_step * (step - start_step) / max(elapsed, 1e-9)
            log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                          f"{lr:.2e}", f"{tps:.0f}", f"{elapsed:.1f}"])
            log_f.flush()
            print(
                f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.1f} min",
                flush=True,
            )
            best_val = min(best_val, losses["val"])

            if step % args.sample_every == 0 or step == args.steps:
                txt = sample_text(model, data, prompt="ROMEO:", n=300)
                with open(os.path.join(args.out, "samples.txt"), "a") as f:
                    f.write(f"\n===== step {step} | val {losses['val']:.4f} =====\n{txt}\n")
                print(f"--- sample @ {step} ---\n{txt[:400]}\n---", flush=True)

            if step % args.ckpt_every == 0 or step == args.steps:
                save(step, losses["val"])
                print(f"saved checkpoint @ {step}", flush=True)

        if step == args.steps:
            break

        x, y = get_batch(data.train, cfg.block_size, args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            tps = tokens_per_step * (step - start_step + 1) / max(elapsed, 1e-9)
            print(f"  step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} "
                  f"| {tps:.0f} tok/s | {elapsed/60:.1f} min", flush=True)

        if args.time_budget and (time.time() - t0) > args.time_budget:
            save(step, float("nan"))
            print(f"time budget reached at step {step}; checkpointed", flush=True)
            break

    log_f.close()
    save(args.steps, best_val)
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
