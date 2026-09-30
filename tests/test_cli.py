import csv
import json

import torch

import evaluate_lpips
import train
from scramblemix import ScrambleMix, lpips_model


def test_train_cli_smoke_resume_and_eval(tmp_path):
    out = tmp_path / "run"
    common = ["--dataset", "fake", "--model", "resnet18", "--batch-size", "8", "--fake-train-size", "8",
              "--fake-test-size", "8", "--views", "2", "--tta", "2", "--workers", "0", "--device", "cpu",
              "--final-repeats", "1", "--out-dir", str(out)]
    train.main(common + ["--epochs", "1"])
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert len(rows) == 1 and float(rows[0]["train_loss"]) > 0
    final = json.load(open(out / "final.json"))
    assert final["epochs_trained"] == 1 and 0 <= final["test_acc_tta_mean"] <= 100
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 1 and ckpt["args"]["views"] == 2
    assert ScrambleMix.load(out / "keys.pt").num_pairs == 4

    train.main(common + ["--epochs", "2", "--resume"])  # continue the same run
    assert len(list(csv.DictReader(open(out / "metrics.csv")))) == 2

    result = train.main(["--eval-only", "--resume", str(out / "last.pt"), "--tta", "4", "--workers", "0",
                         "--device", "cpu", "--final-repeats", "1", "--batch-size", "8"])
    assert result["args"]["model"] == "resnet18" and (out / "eval.json").exists()


def test_evaluate_lpips_cli(tmp_path, monkeypatch):
    # random LPIPS backbone: no weight download in the test
    monkeypatch.setattr(evaluate_lpips, "lpips_model", lambda net, device: lpips_model(net, "cpu", False))
    results = evaluate_lpips.main(["--dataset", "fake", "--num-images", "8", "--batch-size", "4", "--device", "cpu",
                                   "--out-dir", str(tmp_path)])
    assert len(results) == 6 and all(r["num_images"] == 8 for r in results.values())
    assert (tmp_path / "lpips.json").exists() and (tmp_path / "lpips.csv").exists()
