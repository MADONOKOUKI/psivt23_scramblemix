#!/usr/bin/env python
"""Train a classifier on ScrambleMix images and report single-key (T = 1) and multi-key TTA accuracy.

The defaults are the paper setting of the original code (archive/scripts/scramblemix/main_paper.sh
and cifar10.py / trainer.py / dataloader.py):

* f(x; k): pixel-based encryption with the original hard-coded keys, four key pairs;
* D = 4 ScrambleMix views per training image, m ~ Beta(5e-3, 5e-3), a fresh m per view;
* loss = mean cross-entropy over the views + 1.0 x self-teaching loss;
* SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch 256, 200 epochs, lr x 0.1 at epochs 60/120/180;
* evaluation: one view with a random key pair (T = 1, slide 39) and the average of the logits of the
  four key pairs (T = 4, slide 40).

Examples:
    python train.py --dataset cifar10 --model shakedrop              # paper setting
    python train.py --dataset fake --model resnet18 --epochs 1       # smoke test, no download
    python train.py --eval-only --resume runs/cifar10_shakedrop/last.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from scramblemix import EvalViews, ScrambleMix, ScrambleMixViews, average_predictions, scramblemix_loss, self_teaching_loss
from scramblemix.data import NUM_CLASSES, build_datasets, train_augmentation
from scramblemix.keys import ORIGINAL_KEY_DATASETS, consecutive_pairs, load_le_keys, original_pe_keys, random_keys
from scramblemix.models import MODELS, build_model

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScrambleMix training (PSIVT 2023)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="cifar10", choices=sorted(NUM_CLASSES))
    g.add_argument("--data-root", default="./data", help="torchvision datasets are downloaded here")
    g.add_argument("--fake-train-size", type=int, default=512, help="images in the synthetic 'fake' train set")
    g.add_argument("--fake-test-size", type=int, default=256, help="images in the synthetic 'fake' test set")
    g.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    g = p.add_argument_group("ScrambleMix")
    g.add_argument("--scheme", default="pe", choices=["pe", "le"],
                   help="scrambling f(x;k): pixel-based encryption (paper) or block-wise learnable encryption")
    g.add_argument("--keys", default="original", choices=["original", "random"],
                   help="'original': keys of the original code (PE: per dataset, CIFAR-10 keys for 'fake'; "
                        "LE: --le-key-dir); 'random': new keys from --key-seed")
    g.add_argument("--key-seed", type=int, default=0, help="seed of the key set when --keys random")
    g.add_argument("--le-key-dir", default=str(ROOT / "archive" / "utils" / "key4"),
                   help="original LE key files <i>_.pkl (used with --scheme le --keys original)")
    g.add_argument("--channel-mode", default="original", choices=["original", "permutation"],
                   help="PE colour shuffle: exactly as the original code, or true channel permutations")
    g.add_argument("--num-pairs", type=int, default=4, help="key pairs in the key set (original code: 4)")
    g.add_argument("--views", type=int, default=4,
                   help="D: ScrambleMix views (key pairs) per training image (num_of_TTA in the original code)")
    g.add_argument("--tta", type=int, default=4, help="T: key pairs averaged at test time")
    g.add_argument("--alpha", type=float, default=5e-3, help="m ~ Beta(alpha, alpha)")
    g.add_argument("--lambda-st", type=float, default=1.0, help="weight of the self-teaching loss (0 = off)")
    g.add_argument("--tta-reduction", default="logits", choices=["logits", "probs"],
                   help="average logits (original code) or posteriors (slide 37)")
    g.add_argument("--no-mix", action="store_true",
                   help="ablation/baseline: plain image scrambling, one key per view and no mixing")

    g = p.add_argument_group("optimisation")
    g.add_argument("--model", default="shakedrop", choices=sorted(MODELS))
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--milestones", type=int, nargs="+", default=[60, 120, 180])
    g.add_argument("--gamma", type=float, default=0.1)

    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=130, help="the original code seeds torch with 130")
    g.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    g.add_argument("--out-dir", default=None, help="default: runs/<dataset>_<model>_<scheme>_D<views>_T<tta>")
    g.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="checkpoint to resume from ('auto' or no value: <out-dir>/last.pt)")
    g.add_argument("--eval-only", action="store_true", help="only evaluate the --resume checkpoint")
    g.add_argument("--final-repeats", type=int, default=10,
                   help="evaluations after training (views are random); the original code ran 10")
    args = p.parse_args(argv)
    if args.eval_only and args.resume in (None, "auto"):
        p.error("--eval-only needs --resume <checkpoint>")
    return args


# arguments that define a trained model; taken from the checkpoint when resuming / evaluating
MODEL_ARGS = ("dataset", "model", "scheme", "keys", "key_seed", "channel_mode", "num_pairs", "views", "alpha", "no_mix")


def finalize_args(args, saved=None) -> None:
    if saved is not None:
        for k in MODEL_ARGS:
            setattr(args, k, saved[k])
    if args.out_dir is None:
        tag = "nomix" if args.no_mix else "scramblemix"
        args.out_dir = str(Path("runs") / f"{args.dataset}_{args.model}_{tag}_{args.scheme}_D{args.views}_T{args.tta}")
    if not 1 <= args.views <= args.num_pairs or not 1 <= args.tta <= args.num_pairs:
        raise SystemExit(f"--views and --tta must be between 1 and --num-pairs ({args.num_pairs})")
    if args.tta > args.views:
        print(f"note: --tta {args.tta} > --views {args.views}: test-time augmentation also uses key pairs that "
              "were not used for training (as the original code did when num_of_TTA < 4)")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_keyset(args) -> ScrambleMix:
    n_keys = 2 * args.num_pairs
    if args.keys == "original":
        if args.scheme == "pe":
            source = args.dataset if args.dataset in ORIGINAL_KEY_DATASETS else "cifar10"
            keys = original_pe_keys(source, args.channel_mode)
        else:
            if not Path(args.le_key_dir).is_dir():
                raise SystemExit(f"LE key directory {args.le_key_dir} not found; use --keys random")
            keys = load_le_keys(args.le_key_dir, range(n_keys))
        if n_keys > len(keys):
            raise SystemExit(f"the original key set has {len(keys)} keys ({len(keys) // 2} pairs); "
                             f"use --num-pairs <= {len(keys) // 2} or --keys random")
        keys = keys[:n_keys]
    else:
        keys = random_keys(n_keys, args.scheme, 32, args.key_seed, args.channel_mode)
    pairs = [(k, k) for k in keys[::2]] if args.no_mix else consecutive_pairs(keys)
    return ScrambleMix(pairs, alpha=args.alpha)


def make_loaders(args, keyset: ScrambleMix, device: torch.device):
    train_tf = transforms.Compose([train_augmentation(),
                                   ScrambleMixViews(keyset.subset(range(keyset.num_pairs), seed=args.seed), args.views)])
    test_tf = EvalViews(keyset.subset(range(keyset.num_pairs), seed=args.seed + 1), args.views, args.tta)
    train_set, test_set, num_classes = build_datasets(args.dataset, args.data_root, train_tf, test_tf,
                                                      fake_train_size=args.fake_train_size,
                                                      fake_test_size=args.fake_test_size)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=False, **kw)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, **kw)
    return train_loader, test_loader, num_classes


def train_one_epoch(model, loader, optimizer, device, lam: float) -> dict:
    model.train()
    sums = {"loss": 0.0, "st": 0.0}
    correct = seen = images = 0
    for views, target in loader:  # views: [B, D, 3, H, W]
        views = views.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # one forward pass per view, as in the original trainer (BatchNorm/ShakeDrop see one key pair at a time)
        logits = torch.stack([model(views[:, d]) for d in range(views.shape[1])])  # [D, B, K]
        loss = scramblemix_loss(logits, target, lam=lam)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        b = target.size(0)
        sums["loss"] += loss.item() * b
        sums["st"] += self_teaching_loss(logits.detach()).item() * b
        correct += (logits.argmax(-1) == target).sum().item()
        seen += logits.shape[0] * b
        images += b
    return {"train_loss": sums["loss"] / images, "train_st": sums["st"] / images,
            "train_acc": 100.0 * correct / seen}


@torch.no_grad()
def evaluate(model, loader, device, reduction: str = "logits") -> dict:
    model.eval()
    correct_1 = correct_t = n = 0
    for (single, tta), target in loader:
        target = target.to(device, non_blocking=True)
        correct_1 += (model(single.to(device, non_blocking=True)).argmax(1) == target).sum().item()
        pred = average_predictions(model, tta.to(device, non_blocking=True), reduction)
        correct_t += (pred.argmax(1) == target).sum().item()
        n += target.size(0)
    return {"test_acc_single": 100.0 * correct_1 / n, "test_acc_tta": 100.0 * correct_t / n}


def save_checkpoint(path: Path, model, optimizer, epoch: int, keyset: ScrambleMix, args, history) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "keys": keyset.state_dict(), "args": vars(args), "history": history}, str(tmp))
    os.replace(tmp, path)


def main(argv=None) -> dict:
    args = parse_args(argv)
    ckpt = None
    if args.resume not in (None, "auto"):  # explicit checkpoint: it defines the model and the keys
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint {ckpt_path} not found")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if args.out_dir is None:
            args.out_dir = str(ckpt_path.parent)
    finalize_args(args, None if ckpt is None else ckpt["args"])
    out_dir = Path(args.out_dir)
    if args.resume == "auto" and (out_dir / "last.pt").exists():  # same command again: continue the run
        ckpt = torch.load(str(out_dir / "last.pt"), map_location="cpu", weights_only=True)
        finalize_args(args, ckpt["args"])
    if ckpt is not None:
        print(f"loaded checkpoint (epoch {ckpt['epoch']}) from {args.resume if args.resume != 'auto' else out_dir / 'last.pt'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # the key set is part of the model: reuse the checkpoint's keys when resuming
    keyset = ScrambleMix.from_state_dict(ckpt["keys"]) if ckpt is not None else build_keyset(args)
    keyset.save(out_dir / "keys.pt")
    train_loader, test_loader, num_classes = make_loaders(args, keyset, device)

    model = build_model(args.model, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch, history = 0, []
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, history = int(ckpt["epoch"]), list(ckpt["history"])
        scheduler.last_epoch = start_epoch  # the optimizer state already holds the decayed learning rate
        scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]

    print(f"device={device} dataset={args.dataset} model={args.model} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params) "
          f"keys={keyset} D={args.views} T={args.tta} lambda_st={args.lambda_st} out={out_dir}")
    csv_path = out_dir / "metrics.csv"
    fields = ["epoch", "lr", "train_loss", "train_st", "train_acc", "test_acc_single", "test_acc_tta", "seconds"]
    if not args.eval_only:
        if start_epoch == 0 or not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch + 1, "lr": lr}
            row.update(train_one_epoch(model, train_loader, optimizer, device, args.lambda_st))
            scheduler.step()
            row.update(evaluate(model, test_loader, device, args.tta_reduction))
            row["seconds"] = round(time.time() - t0, 2)
            history.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            save_checkpoint(out_dir / "last.pt", model, optimizer, epoch + 1, keyset, args, history)
            print(f"epoch {epoch + 1:3d}/{args.epochs} lr {lr:.4f} loss {row['train_loss']:.4f} "
                  f"(st {row['train_st']:.4f}) train {row['train_acc']:.2f} | test T=1 {row['test_acc_single']:.2f} "
                  f"T={args.tta} {row['test_acc_tta']:.2f} | {row['seconds']:.1f}s", flush=True)

    # views are random (key pair and m), so the final model is evaluated several times like the original code
    runs = [evaluate(model, test_loader, device, args.tta_reduction) for _ in range(max(1, args.final_repeats))]
    single = np.array([r["test_acc_single"] for r in runs])
    tta = np.array([r["test_acc_tta"] for r in runs])
    result = {
        "args": vars(args),
        "epochs_trained": len(history),
        "test_acc_single_mean": float(single.mean()), "test_acc_single_std": float(single.std()),
        "test_acc_tta_mean": float(tta.mean()), "test_acc_tta_std": float(tta.std()),
        "final_runs": runs,
        # the original code printed this "best acc" (maximum over epochs of the single-view test accuracy)
        "best_epoch_test_acc_single": max((h["test_acc_single"] for h in history), default=None),
    }
    name = "eval.json" if args.eval_only else "final.json"
    with open(out_dir / name, "w") as f:
        json.dump(result, f, indent=2)
    print(f"final ({len(runs)} evaluations): T=1 {single.mean():.2f} +- {single.std():.2f} | "
          f"T={args.tta} {tta.mean():.2f} +- {tta.std():.2f}  -> {out_dir / name}")
    return result


if __name__ == "__main__":
    main()
