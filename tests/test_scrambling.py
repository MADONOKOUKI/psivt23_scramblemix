import json
import pickle
import re

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import ARCHIVE, ORIGINAL_CODE, ROOT, load_archived_module
from scramblemix import LearnableEncryption, PixelEncryption, original_pe_keys
from scramblemix.scrambling import PE_ORIGINAL_CHANNEL_MAP, PE_PERMUTATIONS


def rand_images(n, size=32, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (n, 3, size, size), dtype=np.uint8))


# ----------------------------------------------------------------------------- pixel-based encryption
def test_pe_shape_dtype_and_key_seed():
    x = rand_images(2)
    y = PixelEncryption.random(32, seed=0)(x)
    assert y.shape == x.shape and y.dtype == torch.uint8
    assert torch.equal(y, PixelEncryption.random(32, seed=0)(x))
    assert not torch.equal(y, PixelEncryption.random(32, seed=1)(x))


def test_pe_accepts_pil_numpy_and_float():
    key = PixelEncryption.random(32, seed=0)
    x = rand_images(1)[0]
    hwc = x.permute(1, 2, 0).numpy()
    expected = key(x)
    assert torch.equal(key(Image.fromarray(hwc)), expected)
    assert torch.equal(key(hwc), expected)
    assert torch.equal(key(x.float() / 255), expected)


@pytest.mark.parametrize("size", [32, (24, 40)])
def test_pe_permutation_mode_is_invertible(size):
    key = PixelEncryption.random(size, seed=3, channel_mode="permutation")
    h, w = (size, size) if isinstance(size, int) else size
    x = torch.randint(0, 256, (4, 3, h, w), dtype=torch.uint8)
    y = key(x)
    assert not torch.equal(y, x)
    assert torch.equal(key.inverse(y), x)


def test_pe_original_mode_is_not_invertible():
    with pytest.raises(ValueError):
        PixelEncryption.random(32, seed=0).inverse(rand_images(1))


def test_pe_rejects_wrong_size():
    with pytest.raises(ValueError):
        PixelEncryption.random(32, seed=0)(rand_images(1, size=16))


def test_original_channel_map_is_what_the_archived_code_computes():
    orig = load_archived_module("scripts/scramblemix/pixel_based_encryption.py", "orig_pe")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 10, 20, 30
    keep = np.ones(32 * 32 * 3, dtype=np.int64)  # no negative-positive transform
    for code in range(6):
        out = orig.pixel_based_encryption(img.copy(), keep, np.full(32 * 32, code))
        channels = [int(v) // 10 - 1 for v in out[0, 0]]
        assert channels == PE_ORIGINAL_CHANNEL_MAP[code].tolist()
    # only code 0 is a permutation in the original implementation
    assert all(len(set(PE_ORIGINAL_CHANNEL_MAP[c])) < 3 for c in range(1, 6))
    assert all(sorted(p) == [0, 1, 2] for p in PE_PERMUTATIONS.tolist())


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100", "svhn"])
def test_pe_original_mode_matches_archived_code_bit_for_bit(dataset):
    orig = load_archived_module("scripts/scramblemix/pixel_based_encryption.py", "orig_pe")
    strings = json.loads((ROOT / "scramblemix" / "resources" / "original_pe_keys.json").read_text())["keys"][dataset]
    rng = np.random.default_rng(1)
    for key, s in zip(original_pe_keys(dataset), strings):
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        invs = np.array([int(c) for c in s["invs"]])
        colors = np.array([int(c) for c in s["colors"]])
        reference = orig.pixel_based_encryption(img.copy(), invs, colors)  # float64 [H, W, 3] in [0, 255]
        assert np.array_equal(key(img).permute(1, 2, 0).numpy().astype(np.float64), reference)


def test_original_key_file_is_a_verbatim_copy_of_the_archived_source():
    src = ORIGINAL_CODE / "cifar10.py"
    if not src.exists():
        pytest.skip("archive/ not available")
    text = src.read_text()
    stored = json.loads((ROOT / "scramblemix" / "resources" / "original_pe_keys.json").read_text())["keys"]
    sections = re.split(r'\n(?:if|elif) args\.dataset == "(cifar10|cifar100|svhn)":\n', text)
    for name, body in zip(sections[1::2], sections[2::2]):
        invs = re.findall(r'^\s+invs="([01]+)"', body, flags=re.M)
        colors = re.findall(r'^\s+colors="([0-5]+)"', body, flags=re.M)
        assert [k["invs"] for k in stored[name]] == invs
        assert [k["colors"] for k in stored[name]] == colors
    assert sorted(stored) == ["cifar10", "cifar100", "svhn"]


# ----------------------------------------------------------------------------- learnable encryption
def test_le_invertible_and_seeded():
    x = rand_images(3)
    key = LearnableEncryption.random(4, seed=0)
    y = key(x)
    assert y.shape == x.shape and y.dtype == torch.uint8
    assert torch.equal(key.inverse(y), x)
    assert torch.equal(y, LearnableEncryption.random(4, seed=0)(x))
    assert int(key.flip.sum()) == 47  # strict '>' as in the reference code


def test_le_matches_archived_block_scramble():
    ble = load_archived_module("scripts/scramblemix/learnable_encryption_augmix.py", "orig_le")
    rng = np.random.default_rng(2)
    key_files = [ARCHIVE / "utils" / "key4" / f"{i}_.pkl" for i in (0, 5, 63)]
    for path in key_files:
        ref = ble.BlockScramble(str(path))
        ours = LearnableEncryption.from_pickle(path)
        x = rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        expected = ref.doScramble(x.copy(), ref.key, ref.rev)
        got = ours(torch.from_numpy(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        assert np.array_equal(got, expected)
    # random keys too, including the inverse (Decramble)
    key = LearnableEncryption.random(4, seed=7)
    ref = ble.BlockScramble([4, 4, 3])
    ref.setKey(key.key.astype(np.uint32))
    x = rng.integers(0, 256, (1, 32, 32, 3), dtype=np.uint8)
    got = key(torch.from_numpy(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
    assert np.array_equal(got, ref.doScramble(x.copy(), ref.key, ref.rev))
    assert np.array_equal(ref.doScramble(got.copy(), ref.invKey, ref.rev), x)


def test_le_pickle_loader_refuses_arbitrary_objects(tmp_path):
    path = tmp_path / "evil.pkl"
    with open(path, "wb") as f:
        pickle.dump([[4, 4, 3], {"not": "an array"}, Image.new("RGB", (1, 1))], f)
    with pytest.raises(pickle.UnpicklingError):
        LearnableEncryption.from_pickle(path)


def test_le_rejects_indivisible_size():
    with pytest.raises(ValueError):
        LearnableEncryption.random(4, seed=0)(rand_images(1, size=30))


def test_state_dict_roundtrip():
    from scramblemix.scrambling import scrambler_from_state_dict

    x = rand_images(2)
    for key in (PixelEncryption.random(32, seed=0), PixelEncryption.random(32, seed=0, channel_mode="permutation"),
                LearnableEncryption.random(4, seed=0)):
        assert torch.equal(scrambler_from_state_dict(key.state_dict())(x), key(x))
