#!/usr/bin/env python
"""Visual information hiding: LPIPS between original and scrambled test images (higher = better hiding).

Evaluates, on the same test images,
  * PE          -- pixel-based encryption with a single key,
  * LE          -- block-wise learnable encryption (4x4 blocks) with a single key,
  * ScrambleMix -- one view with a random key pair and m ~ Beta(alpha, alpha) (the T = 1 test view),
  * ScrambleMix with fixed mixing ratios m (default 0.1 / 0.3 / 0.5, the values shown on slide 42).
LPIPS uses AlexNet features (lpips.LPIPS(net='alex')), like the original code.

Examples:
    python evaluate_lpips.py --dataset cifar10                    # full CIFAR-10 test set, original keys
    python evaluate_lpips.py --keyset runs/<run>/keys.pt          # the key set of a training run
    python evaluate_lpips.py --dataset fake --num-images 64       # smoke test (no dataset download)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scramblemix import LearnableEncryption, ScrambleMix, lpips_distance, lpips_model
from scramblemix._image import as_uint8_tensor
from scramblemix.data import NUM_CLASSES, build_datasets
from scramblemix.keys import ORIGINAL_KEY_DATASETS

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="cifar10", choices=sorted(NUM_CLASSES))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--num-images", type=int, default=None, help="first N test images (default: all)")
    p.add_argument("--keys", default="original", choices=["original", "random"],
                   help="original keys of the paper's code, or random keys from --key-seed")
    p.add_argument("--key-seed", type=int, default=0)
    p.add_argument("--keyset", default=None, help="ScrambleMix key set saved by train.py (keys.pt); overrides --keys")
    p.add_argument("--le-key", default=str(ROOT / "archive" / "utils" / "key4" / "0_.pkl"),
                   help="original LE key file (used with --keys original)")
    p.add_argument("--alpha", type=float, default=5e-3)
    p.add_argument("--mix-ratios", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    p.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0, help="seed of the sampled key pairs and mixing ratios")
    p.add_argument("--out-dir", default=None, help="default: runs/lpips_<dataset>")
    return p.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_methods(args):
    """name -> function(uint8 images [B, 3, H, W]) -> float scrambled images in [0, 1]."""
    if args.keyset:
        sm = ScrambleMix.load(args.keyset, seed=args.seed)
    elif args.keys == "original":
        sm = ScrambleMix.original(args.dataset if args.dataset in ORIGINAL_KEY_DATASETS else "cifar10",
                                  alpha=args.alpha, seed=args.seed)
    else:
        sm = ScrambleMix.random(4, "pe", 32, key_seed=args.key_seed, alpha=args.alpha, seed=args.seed)
    if args.keys == "original" and Path(args.le_key).exists():
        le = LearnableEncryption.from_pickle(args.le_key)
    else:
        le = LearnableEncryption.random(4, seed=args.key_seed)
    first_key = sm.pairs[0][0]
    methods = {
        f"{sm.scheme.upper()} (single key)": lambda x: first_key(x).float() / 255.0,
        "LE (single key)": lambda x: le(x).float() / 255.0,
        f"ScrambleMix (m ~ Beta({sm.alpha:g}, {sm.alpha:g}), random key pair)": lambda x: sm.sample(x),
    }
    for r in args.mix_ratios:
        methods[f"ScrambleMix (m = {r:g}, key pair 0)"] = (lambda r: lambda x: sm.mix(x, 0, r))(r)
    return methods


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir or Path("runs") / f"lpips_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, test_set, _ = build_datasets(args.dataset, args.data_root, test_transform=as_uint8_tensor)
    if args.num_images is not None:
        test_set = Subset(test_set, range(min(args.num_images, len(test_set))))
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    methods = build_methods(args)
    model = lpips_model(args.net, device)
    scores = {name: [] for name in methods}
    for x, _ in loader:
        original = x.float() / 255.0
        for name, scramble in methods.items():
            scores[name].append(lpips_distance(original, scramble(x), model=model, device=device))

    results = {}
    for name, parts in scores.items():
        v = torch.cat(parts).numpy()
        results[name] = {"lpips_mean": float(v.mean()), "lpips_std": float(v.std()), "num_images": int(v.size)}
    with open(out_dir / "lpips.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    with open(out_dir / "lpips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "lpips_mean", "lpips_std", "num_images"])
        for name, r in results.items():
            w.writerow([name, f"{r['lpips_mean']:.4f}", f"{r['lpips_std']:.4f}", r["num_images"]])
    width = max(len(n) for n in results)
    print(f"LPIPS ({args.net}) between original and scrambled {args.dataset} test images (higher = better hiding)")
    for name, r in results.items():
        print(f"  {name:<{width}}  {r['lpips_mean']:.4f} +- {r['lpips_std']:.4f}  (n={r['num_images']})")
    print(f"-> {out_dir / 'lpips.json'}")
    return results


if __name__ == "__main__":
    main()
