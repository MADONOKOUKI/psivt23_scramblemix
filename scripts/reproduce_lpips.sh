#!/usr/bin/env bash
# Visual information hiding measured with LPIPS (AlexNet) between original and scrambled test images,
# for single-key PE and LE scrambling and for ScrambleMix (sampled m and m = 0.1 / 0.3 / 0.5).
# The paper evaluates LPIPS; its values are not in the public slides. Results: runs/lpips_<dataset>/lpips.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY evaluate_lpips.py --dataset cifar10  --out-dir runs/lpips_cifar10
$PY evaluate_lpips.py --dataset cifar100 --out-dir runs/lpips_cifar100
$PY evaluate_lpips.py --dataset svhn     --out-dir runs/lpips_svhn
