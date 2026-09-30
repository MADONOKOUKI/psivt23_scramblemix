#!/usr/bin/env bash
# A few-minute end-to-end check on synthetic data (no dataset download): training + TTA evaluation + LPIPS.
# FakeData labels are random, so accuracy stays at chance level (~10 %); this only checks the pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY train.py --dataset fake --model resnet18 --epochs 2 --milestones 1 --fake-train-size 256 \
    --fake-test-size 128 --batch-size 64 --final-repeats 2 --out-dir runs/smoke
$PY evaluate_lpips.py --dataset fake --num-images 64 --out-dir runs/smoke_lpips
