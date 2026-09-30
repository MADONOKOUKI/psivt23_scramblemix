import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scramblemix import EvalViews, ScrambleMix, ScrambleMixViews


def pil_image(seed=0, size=32):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


def test_accepts_pil_numpy_and_tensors():
    sm = ScrambleMix.random(num_pairs=2, key_seed=0, mix_ratio=0.3)
    img = pil_image()
    arr = np.asarray(img)  # read-only view of the PIL image
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    ref = sm.mix(img, pair=1)
    assert ref.shape == (3, 32, 32) and ref.dtype == torch.float32
    assert 0.0 <= ref.min() and ref.max() <= 1.0
    for x in (arr, arr.astype(np.float64) / 255, t, t.float() / 255):
        assert torch.allclose(sm.mix(x, pair=1), ref)
    batch = sm(torch.stack([t, t, t]))
    assert batch.shape == (3, 3, 32, 32)


def test_mix_formula_and_endpoints():
    sm = ScrambleMix.random(num_pairs=3, key_seed=1)
    img = pil_image(1)
    k1, k2 = sm.pairs[2]
    a, b = k1(img).float() / 255, k2(img).float() / 255
    assert torch.allclose(sm.mix(img, 2, 0.0), a)
    assert torch.allclose(sm.mix(img, 2, 1.0), b)
    assert torch.allclose(sm.mix(img, 2, 0.25), 0.75 * a + 0.25 * b, atol=1e-6)


def test_seeded_determinism():
    img = pil_image(2)
    outs = []
    for _ in range(2):
        sm = ScrambleMix.random(num_pairs=4, key_seed=0, seed=123)
        outs.append(torch.stack([sm(img) for _ in range(5)]))
    assert torch.equal(outs[0], outs[1])
    other = ScrambleMix.random(num_pairs=4, key_seed=0, seed=124)
    assert not torch.equal(outs[0], torch.stack([other(img) for _ in range(5)]))


def test_different_key_seeds_give_different_keys():
    img = pil_image(3)
    a = ScrambleMix.random(num_pairs=1, key_seed=0).mix(img, 0, 0.5)
    b = ScrambleMix.random(num_pairs=1, key_seed=1).mix(img, 0, 0.5)
    assert not torch.equal(a, b)


def test_mixing_ratio_distribution_of_the_paper_setting():
    sm = ScrambleMix.random(num_pairs=1, key_seed=0, alpha=5e-3, seed=0)
    m = sm.sample_mix_ratio(20000)
    assert m.min() >= 0 and m.max() <= 1
    assert abs(m.mean() - 0.5) < 0.03                  # symmetric Beta(a, a)
    assert np.mean((m > 0.01) & (m < 0.99)) < 0.05       # alpha = 5e-3: m is almost always ~0 or ~1


def test_views_shapes_and_pairs():
    sm = ScrambleMix.random(num_pairs=4, key_seed=0, mix_ratio=0.0)
    img = pil_image(4)
    v = sm.views(img)
    assert v.shape == (4, 3, 32, 32)
    for p in range(4):  # with m = 0 every view is f(x; k1) of its own pair
        assert torch.allclose(v[p], sm.pairs[p][0](img).float() / 255)
    vb = sm.views(torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1)] * 2), pairs=[0, 2])
    assert vb.shape == (2, 2, 3, 32, 32)
    assert ScrambleMixViews(sm, 3)(img).shape == (3, 3, 32, 32)


def test_torchvision_compose_and_eval_views():
    sm = ScrambleMix.original("cifar10", seed=0)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   ScrambleMixViews(sm, num_views=4)])
    assert train_tf(pil_image(5)).shape == (4, 3, 32, 32)
    single, tta = EvalViews(sm, num_train_views=2, num_tta_views=4)(pil_image(5))
    assert single.shape == (3, 32, 32) and tta.shape == (4, 3, 32, 32)


def test_learnable_encryption_scheme():
    sm = ScrambleMix.random(num_pairs=2, scheme="le", key_seed=0)
    assert sm.scheme == "le"
    assert sm(pil_image(6)).shape == (3, 32, 32)


def test_save_load_roundtrip(tmp_path):
    sm = ScrambleMix.original("svhn", alpha=0.2)
    sm.save(tmp_path / "keys.pt")
    loaded = ScrambleMix.load(tmp_path / "keys.pt")
    assert loaded.num_pairs == 4 and loaded.alpha == 0.2
    img = pil_image(7)
    for p in range(4):
        assert torch.equal(loaded.mix(img, p, 0.4), sm.mix(img, p, 0.4))
    # identical keys inside a pair (plain-scrambling baseline) survive the round trip
    k = sm.pairs[0][0]
    same = ScrambleMix([(k, k)])
    same.save(tmp_path / "same.pt")
    assert torch.equal(ScrambleMix.load(tmp_path / "same.pt").mix(img, 0, 0.7), k(img).float() / 255)


def test_worker_streams_are_independent(monkeypatch):
    class Info:
        def __init__(self, seed):
            self.seed, self.id = seed, 0

    sm = ScrambleMix.random(num_pairs=2, key_seed=0, seed=5)
    draws = []
    for worker_seed, pid in [(1000, 101), (1001, 102)]:
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda s=worker_seed: Info(s))
        monkeypatch.setattr(os, "getpid", lambda p=pid: p)
        draws.append(sm.sample_mix_ratio(16))
    assert not np.allclose(draws[0], draws[1])


def test_dataloader_with_workers():
    sm = ScrambleMix.random(num_pairs=2, key_seed=0)
    ds = datasets.FakeData(8, (3, 32, 32), 10, transform=ScrambleMixViews(sm))
    views, labels = next(iter(DataLoader(ds, batch_size=8, num_workers=2)))
    assert views.shape == (8, 2, 3, 32, 32) and labels.shape == (8,)


def test_matches_archived_augmix_dataset(original_augmix_module, monkeypatch):
    """End to end against the original data pipeline (cifar10.py AugMixDataset) with m fixed to 0.3."""
    module = original_augmix_module
    monkeypatch.setattr(module.np.random, "seed", lambda *a, **k: None)
    monkeypatch.setattr(module.np.random, "beta", lambda a, b: 0.3)
    img = pil_image(8)
    out = module.AugMixDataset([(img, 3)], transforms.ToTensor(), True)[0]
    mixed = torch.stack(out[1:5])  # four training views, values in [0, 255] in the original code
    ours = ScrambleMix.original("cifar10", mix_ratio=0.3).views(img)
    assert torch.allclose(ours * 255, mixed, atol=1e-3)
