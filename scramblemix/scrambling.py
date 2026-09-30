"""Key-based image scrambling functions ``f(x; k)`` used by ScrambleMix.

The paper compares two learnable image-encryption ("image scrambling") schemes and builds
ScrambleMix on top of them (slides 17-25 and 38-39 of ``psivt.pdf``):

* :class:`PixelEncryption` -- random pixel-based encryption ("Random PE",
  Sirichoptedumrong et al., 2019): a per-pixel negative-positive transform followed by a
  per-pixel colour-channel shuffle. This is the ``f(x; k)`` used by ScrambleMix in the
  original code (``archive/scripts/scramblemix/pixel_based_encryption.py``).
* :class:`LearnableEncryption` -- block-wise learnable encryption ("LE", Tanaka,
  ICCE-TW 2018): every ``M x M`` block is split into 4-bit nibbles that are shuffled and
  partially negated with a key (``archive/scripts/scramblemix/learnable_encryption_augmix.py``,
  based on Tanaka's reference implementation https://github.com/mastnk/ICCE-TW2018).

Every scrambler is a callable mapping an 8-bit image (PIL image, ``[H, W, 3]`` numpy array
or ``[..., 3, H, W]`` tensor) to a ``uint8`` tensor ``[..., 3, H, W]``. Keys are plain
numpy arrays; they can be generated from a seed with ``.random()`` and stored with
:func:`scramblemix.transform.ScrambleMix.save`.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from ._image import ImageLike, as_uint8_tensor, pair

SeedLike = Union[None, int, np.random.Generator]

#: The six colour-shuffle codes of pixel-based encryption: output channel ``c`` of a pixel
#: with code ``s`` takes input channel ``PE_PERMUTATIONS[s][c]``.
PE_PERMUTATIONS = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

#: What the original implementation actually computes for each code. It applies the
#: permutation with ``img[:,:,0], img[:,:,1], img[:,:,2] = img[:,:,p0], img[:,:,p1], img[:,:,p2]``;
#: the right-hand side holds numpy *views*, so a channel overwritten by an earlier assignment
#: is read back. Only code 0 (the identity) is a permutation; codes 1-5 duplicate a channel.
PE_ORIGINAL_CHANNEL_MAP = np.array([[0, 1, 2], [0, 2, 2], [1, 1, 2], [1, 2, 1], [2, 2, 2], [2, 1, 2]])

CHANNEL_MODES = ("original", "permutation")


def _rng(seed: SeedLike) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _check_size(x: torch.Tensor, expected: Tuple[int, int], what: str) -> None:
    if tuple(x.shape[-2:]) != tuple(expected):
        raise ValueError(
            f"{what} key was generated for {expected[0]}x{expected[1]} images but the input is "
            f"{x.shape[-2]}x{x.shape[-1]}; resize/crop the image or create keys with image_size={tuple(x.shape[-2:])}"
        )


class PixelEncryption:
    """Random pixel-based encryption (PE) of Sirichoptedumrong et al. (2019).

    For every pixel ``(h, w)`` the key holds

    * ``negate[h, w, c]`` (bool): apply the negative-positive transform ``v -> 255 - v`` to channel ``c``;
    * ``shuffle[h, w]`` in ``{0, ..., 5}``: which of the six colour-channel shuffles to apply.

    Args:
        negate: boolean array ``[H, W, 3]``.
        shuffle: integer array ``[H, W]`` with values in ``0..5``.
        channel_mode: ``"original"`` (default) reproduces the original ScrambleMix code bit for bit,
            including its in-place channel assignment that duplicates a colour channel for shuffle
            codes 1-5 (see :data:`PE_ORIGINAL_CHANNEL_MAP`); this is what produced the paper's numbers,
            but it is not invertible. ``"permutation"`` applies the six codes as true channel
            permutations (the textbook, invertible pixel-based encryption).

    Note:
        The original code stores the negation mask as ``invs`` with the opposite polarity
        (``1`` = keep, ``0`` = negate); :meth:`from_original_strings` converts it.
    """

    scheme = "pe"

    def __init__(self, negate: np.ndarray, shuffle: np.ndarray, channel_mode: str = "original") -> None:
        negate = np.asarray(negate).astype(bool)
        shuffle = np.asarray(shuffle).astype(np.int64)
        if negate.ndim != 3 or negate.shape[2] != 3:
            raise ValueError(f"negate must have shape [H, W, 3], got {negate.shape}")
        if shuffle.shape != negate.shape[:2]:
            raise ValueError(f"shuffle must have shape {negate.shape[:2]}, got {shuffle.shape}")
        if shuffle.size and (shuffle.min() < 0 or shuffle.max() > 5):
            raise ValueError("shuffle codes must be in 0..5")
        if channel_mode not in CHANNEL_MODES:
            raise ValueError(f"channel_mode must be one of {CHANNEL_MODES}, got {channel_mode!r}")
        self.negate = negate
        self.shuffle = shuffle
        self.channel_mode = channel_mode
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ constructors
    @classmethod
    def random(cls, image_size: Union[int, Tuple[int, int]] = 32, seed: SeedLike = None,
               channel_mode: str = "original") -> "PixelEncryption":
        """Draw a random key for ``image_size`` images (``seed`` makes it reproducible)."""
        h, w = pair(image_size)
        rng = _rng(seed)
        negate = rng.integers(0, 2, size=(h, w, 3)).astype(bool)
        shuffle = rng.integers(0, 6, size=(h, w))
        return cls(negate, shuffle, channel_mode)

    @classmethod
    def from_original_strings(cls, invs: str, colors: str, image_size: Union[int, Tuple[int, int]] = 32,
                              channel_mode: str = "original") -> "PixelEncryption":
        """Build a key from the ``invs``/``colors`` strings hard-coded in the original code."""
        h, w = pair(image_size)
        keep = np.frombuffer(invs.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("0")
        codes = np.frombuffer(colors.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("0")
        if keep.size != h * w * 3 or codes.size != h * w:
            raise ValueError("key strings do not match the requested image size")
        return cls((keep == 0).reshape(h, w, 3), codes.reshape(h, w), channel_mode)

    # ------------------------------------------------------------------ properties
    @property
    def image_size(self) -> Tuple[int, int]:
        return tuple(self.shuffle.shape)  # type: ignore[return-value]

    @property
    def invertible(self) -> bool:
        return self.channel_mode == "permutation"

    def _tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._cache:
            table = PE_ORIGINAL_CHANNEL_MAP if self.channel_mode == "original" else PE_PERMUTATIONS
            negate = torch.from_numpy(self.negate).permute(2, 0, 1).contiguous()          # [3, H, W]
            index = torch.from_numpy(table[self.shuffle]).permute(2, 0, 1).contiguous()  # [3, H, W]
            self._cache[key] = (negate.to(device), index.to(device))
        return self._cache[key]

    # ------------------------------------------------------------------ scrambling
    def __call__(self, img: ImageLike) -> torch.Tensor:
        """Scramble an 8-bit image; returns ``uint8`` ``[..., 3, H, W]``."""
        x = as_uint8_tensor(img)
        _check_size(x, self.image_size, "PixelEncryption")
        negate, index = self._tensors(x.device)
        y = torch.where(negate, 255 - x, x)
        return torch.gather(y, -3, index.expand(x.shape))

    def inverse(self, img: ImageLike) -> torch.Tensor:
        """Undo :meth:`__call__` (only for ``channel_mode="permutation"``)."""
        if not self.invertible:
            raise ValueError(
                "channel_mode='original' duplicates colour channels for shuffle codes 1-5 (exactly like the "
                "original code), so it cannot be inverted; use channel_mode='permutation' for an invertible key"
            )
        y = as_uint8_tensor(img)
        _check_size(y, self.image_size, "PixelEncryption")
        negate, index = self._tensors(y.device)
        z = torch.gather(y, -3, torch.argsort(index, dim=0).expand(y.shape))
        return torch.where(negate, 255 - z, z)

    # ------------------------------------------------------------------ (de)serialisation
    def state_dict(self) -> Dict[str, object]:
        return {
            "scheme": self.scheme,
            "negate": torch.from_numpy(self.negate.astype(np.uint8)),
            "shuffle": torch.from_numpy(self.shuffle.astype(np.uint8)),
            "channel_mode": self.channel_mode,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "PixelEncryption":
        return cls(np.asarray(state["negate"]), np.asarray(state["shuffle"]), str(state["channel_mode"]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __repr__(self) -> str:
        h, w = self.image_size
        return f"PixelEncryption(image_size=({h}, {w}), channel_mode={self.channel_mode!r})"


class LearnableEncryption:
    """Block-wise learnable image encryption (LE) of Tanaka (ICCE-TW 2018).

    The image is divided into ``block_size x block_size`` blocks. Each block (``d = block_size**2 * 3``
    8-bit values, ordered row, column, channel) is split into ``2d`` 4-bit nibbles (the ``d`` low nibbles
    followed by the ``d`` high nibbles). Nibbles at the key-selected positions are negated (``v -> 15 - v``),
    all nibbles are permuted by the key, the same positions are negated again and the nibbles are packed back
    into bytes. Every block uses the same key.

    Args:
        key: permutation of ``0 .. 2d-1``.
        block_size: block size ``M`` (the original code uses 4, key files ``key4/*.pkl``).
        channels: number of colour channels (3).

    Note:
        As in the reference code, the negated positions are ``key > key.size / 2`` (strict inequality,
        i.e. 47 of 96 positions for 4x4x3 blocks); this is kept for exact compatibility.
    """

    scheme = "le"

    def __init__(self, key: np.ndarray, block_size: int = 4, channels: int = 3) -> None:
        key = np.asarray(key).astype(np.int64).ravel()
        d = block_size * block_size * channels
        if key.size != 2 * d or not np.array_equal(np.sort(key), np.arange(2 * d)):
            raise ValueError(f"key must be a permutation of 0..{2 * d - 1} for block_size={block_size}")
        self.key = key
        self.block_size = int(block_size)
        self.channels = int(channels)
        self.flip = key > key.size / 2
        self.inv_key = np.argsort(key)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def random(cls, block_size: int = 4, channels: int = 3, seed: SeedLike = None) -> "LearnableEncryption":
        """Draw a random key (``seed`` makes it reproducible)."""
        return cls(_rng(seed).permutation(2 * block_size * block_size * channels), block_size, channels)

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "LearnableEncryption":
        """Load a key file written by the original code (``[block_size, key]`` pickles,
        e.g. ``archive/utils/key4/0_.pkl``). Only numpy arrays are accepted by the unpickler."""
        with open(path, "rb") as f:
            block, key = _NumpyOnlyUnpickler(f).load()
        bh, bw, channels = (int(v) for v in block)
        if bh != bw:
            raise ValueError(f"only square blocks are supported, got {block}")
        return cls(np.asarray(key), block_size=bh, channels=channels)

    def _tensors(self, device: torch.device):
        k = str(device)
        if k not in self._cache:
            self._cache[k] = (torch.from_numpy(self.flip).to(device),
                              torch.from_numpy(self.key).to(device),
                              torch.from_numpy(self.inv_key).to(device))
        return self._cache[k]

    def _apply(self, x: torch.Tensor, order: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
        *lead, c, h, w = x.shape
        b = self.block_size
        if c != self.channels or h % b or w % b:
            raise ValueError(f"LearnableEncryption expects {self.channels} channels and H, W divisible by {b}; "
                             f"got {tuple(x.shape)}")
        d = b * b * c
        # [..., C, H, W] -> [..., H/b, W/b, b*b*C] (block pixels in row, column, channel order)
        v = x.movedim(-3, -1).reshape(*lead, h // b, b, w // b, b, c).transpose(-4, -3).reshape(*lead, h // b, w // b, d)
        nib = torch.cat([v & 0xF, v >> 4], dim=-1)
        nib = torch.where(flip, 15 - nib, nib)
        nib = nib[..., order]
        nib = torch.where(flip, 15 - nib, nib)
        v = (nib[..., d:] << 4) + nib[..., :d]
        v = v.reshape(*lead, h // b, w // b, b, b, c).transpose(-4, -3).reshape(*lead, h, w, c)
        return v.movedim(-1, -3).contiguous()

    def __call__(self, img: ImageLike) -> torch.Tensor:
        """Scramble an 8-bit image; returns ``uint8`` ``[..., 3, H, W]``."""
        x = as_uint8_tensor(img)
        flip, key, _ = self._tensors(x.device)
        return self._apply(x, key, flip)

    def inverse(self, img: ImageLike) -> torch.Tensor:
        """Undo :meth:`__call__`."""
        y = as_uint8_tensor(img)
        flip, _, inv_key = self._tensors(y.device)
        return self._apply(y, inv_key, flip)

    @property
    def invertible(self) -> bool:
        return True

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        return None  # works for any H, W divisible by block_size

    def state_dict(self) -> Dict[str, object]:
        return {"scheme": self.scheme, "key": torch.from_numpy(self.key.copy()),
                "block_size": self.block_size, "channels": self.channels}

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "LearnableEncryption":
        return cls(np.asarray(state["key"]), int(state["block_size"]), int(state["channels"]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __repr__(self) -> str:
        return f"LearnableEncryption(block_size={self.block_size}, channels={self.channels})"


def scrambler_from_state_dict(state: Dict[str, object]):
    """Rebuild a :class:`PixelEncryption` or :class:`LearnableEncryption` from its ``state_dict()``."""
    scheme = state["scheme"]
    if scheme == "pe":
        return PixelEncryption.from_state_dict(state)
    if scheme == "le":
        return LearnableEncryption.from_state_dict(state)
    raise ValueError(f"unknown scrambling scheme {scheme!r}")


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler for the original ``key4/*.pkl`` files that refuses anything but numpy arrays."""

    _ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
                ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")
        if module.startswith("numpy.core") and hasattr(np, "_core"):  # numpy >= 2 renamed numpy.core
            module = "numpy._core" + module[len("numpy.core"):]
        return super().find_class(module, name)
