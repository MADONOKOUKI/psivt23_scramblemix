"""Key sets: the original keys of the paper's code, random key sets, and key files.

The original code (``archive/scripts/scramblemix/cifar10.py``) hard-codes eight pixel-based
encryption keys per dataset (CIFAR-10, CIFAR-100, SVHN) and pairs them as (0, 1), (2, 3),
(4, 5), (6, 7): a *secret key set* of four key pairs shared by training and test-time
augmentation (slides 29 and 37). The keys are shipped verbatim in
``scramblemix/resources/original_pe_keys.json``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from .scrambling import LearnableEncryption, PixelEncryption, SeedLike, _rng

ORIGINAL_KEY_DATASETS = ("cifar10", "cifar100", "svhn")
_RESOURCES = Path(__file__).resolve().parent / "resources"


@lru_cache(maxsize=None)
def _original_key_strings() -> dict:
    with open(_RESOURCES / "original_pe_keys.json", encoding="utf-8") as f:
        return json.load(f)["keys"]


def original_pe_keys(dataset: str = "cifar10", channel_mode: str = "original") -> List[PixelEncryption]:
    """The eight 32x32 pixel-based encryption keys hard-coded in the original code for ``dataset``."""
    if dataset not in ORIGINAL_KEY_DATASETS:
        raise ValueError(f"original keys exist for {ORIGINAL_KEY_DATASETS}, not {dataset!r}")
    return [PixelEncryption.from_original_strings(k["invs"], k["colors"], 32, channel_mode)
            for k in _original_key_strings()[dataset]]


def load_le_keys(directory: Union[str, Path], indices: Sequence[int] = range(8)) -> List[LearnableEncryption]:
    """Load learnable-encryption keys ``<directory>/<i>_.pkl`` written by the original code
    (``archive/utils/key4`` holds 64 keys for 4x4 blocks)."""
    directory = Path(directory)
    return [LearnableEncryption.from_pickle(directory / f"{i}_.pkl") for i in indices]


def random_keys(num_keys: int, scheme: str = "pe", image_size: Union[int, Tuple[int, int]] = 32,
                seed: SeedLike = None, channel_mode: str = "original", block_size: int = 4) -> list:
    """Draw ``num_keys`` independent random keys (reproducible with ``seed``)."""
    rng = _rng(seed)
    if scheme == "pe":
        return [PixelEncryption.random(image_size, rng, channel_mode) for _ in range(num_keys)]
    if scheme == "le":
        return [LearnableEncryption.random(block_size, 3, rng) for _ in range(num_keys)]
    raise ValueError(f"scheme must be 'pe' or 'le', got {scheme!r}")


def consecutive_pairs(keys: Sequence) -> List[Tuple[object, object]]:
    """Pair keys as (0, 1), (2, 3), ... like the original code."""
    if len(keys) % 2:
        raise ValueError("an even number of keys is needed to form key pairs")
    return [(keys[i], keys[i + 1]) for i in range(0, len(keys), 2)]
