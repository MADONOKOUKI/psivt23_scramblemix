#!/usr/bin/env bash
# ScrambleMix rows of the accuracy tables in the PSIVT 2023 slides (psivt.pdf):
#   slide 39  "Results (T=1, w/o Test-Time Augmentation)"   -> test_acc_single_mean in runs/<run>/final.json
#   slide 40  "Results (T>=1, with Test-Time Augmentation)" -> test_acc_tta_mean    (T = 4 key pairs)
# Both numbers come from the same run. train.py defaults = archive/scripts/scramblemix/main_paper.sh:
# D = 4 views, self-teaching loss (lambda = 1), m ~ Beta(5e-3, 5e-3), the original pixel-based encryption
# keys, SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch 256, 200 epochs, lr x0.1 at 60/120/180.
# Datasets are downloaded to ./data. A CUDA GPU is recommended: every step runs D = 4 forward/backward passes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

# "WideResNet40x10" table. The original code builds WideResNet(depth=40, widen_factor=2);
# pass --model wrn40_10 instead to train the widen-factor-10 network named on the slides.
$PY train.py --dataset cifar10  --model wrn40_2   --out-dir runs/cifar10_wrn40_2
$PY train.py --dataset cifar100 --model wrn40_2   --out-dir runs/cifar100_wrn40_2
$PY train.py --dataset svhn     --model wrn40_2   --out-dir runs/svhn_wrn40_2

# "Shakedrop" table: PyramidNet-110 (alpha = 270) with ShakeDrop.
$PY train.py --dataset cifar10  --model shakedrop --out-dir runs/cifar10_shakedrop
$PY train.py --dataset cifar100 --model shakedrop --out-dir runs/cifar100_shakedrop
$PY train.py --dataset svhn     --model shakedrop --out-dir runs/svhn_shakedrop
