"""The ScrambleMix transform (Madono, Tanaka and Onishi, PSIVT 2023).

ScrambleMix scrambles two copies of an image with the two keys of a key pair and mixes them
(method figure, slides 24-25 of ``psivt.pdf``):

    x~ = (1 - m) * f(x; k_1) + m * f(x; k_2),        m ~ Beta(alpha, alpha)

A *key set* of several key pairs is kept on the edge side. Training uses one ScrambleMix view
per key pair (``D`` views, slide 29) and inference averages the predictions of ``T`` views made
with the same key pairs (ScrambleMix TTA, slide 37). A new ``m`` is drawn for every view.

Defaults follow the original code (``archive/scripts/scramblemix/cifar10.py``): four key pairs
of pixel-based encryption keys and ``alpha = 5e-3``. With such a small ``alpha`` the mixing
ratio is almost always close to 0 or 1 (about 2 % of the draws fall in ``(0.01, 0.99)``), so a
view is usually dominated by one of the two keys of its pair.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ._image import ImageLike, as_uint8_tensor, batched
from .keys import consecutive_pairs, original_pe_keys, random_keys
from .scrambling import scrambler_from_state_dict

KEYSET_FORMAT = "scramblemix-keyset-v1"


class ScrambleMix:
    """ScrambleMix transform.

    Works like a torchvision transform: it takes a PIL image, an ``[H, W, 3]`` numpy array or a
    ``[3, H, W]`` / ``[B, 3, H, W]`` tensor (``uint8``, or float in ``[0, 1]``) and returns a
    ``float32`` tensor in ``[0, 1]`` with the same layout as ``ToTensor`` (so it replaces
    ``ToTensor`` at the end of a PIL pipeline).

    Args:
        pairs: the secret key set, a sequence of key pairs ``(k_1, k_2)`` of scramblers
            (:class:`~scramblemix.PixelEncryption` or :class:`~scramblemix.LearnableEncryption`).
            Use :meth:`random`, :meth:`original` or :meth:`load` to build one.
        alpha: parameter of ``m ~ Beta(alpha, alpha)``; ``5e-3`` in the original code.
        mix_ratio: use this fixed ``m`` instead of sampling it (e.g. ``0.5``).
        seed: seed of the sampler of ``m`` and of the key pair. ``None`` uses fresh OS entropy.
            Inside ``DataLoader`` workers every worker gets its own random stream (derived from
            ``seed`` and the worker seed), so views are never duplicated across workers.
    """

    def __init__(self, pairs: Sequence[Tuple[object, object]], alpha: float = 5e-3,
                 mix_ratio: Optional[float] = None, seed: Optional[int] = None) -> None:
        self.pairs: List[Tuple[object, object]] = [tuple(p) for p in pairs]  # type: ignore[misc]
        if not self.pairs or any(len(p) != 2 for p in self.pairs):
            raise ValueError("pairs must be a non-empty sequence of (key_1, key_2) tuples")
        if mix_ratio is None and not alpha > 0:
            raise ValueError("alpha must be > 0 (or pass a fixed mix_ratio)")
        if mix_ratio is not None and not 0.0 <= mix_ratio <= 1.0:
            raise ValueError("mix_ratio must be in [0, 1]")
        self.alpha = float(alpha)
        self.mix_ratio = None if mix_ratio is None else float(mix_ratio)
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: Optional[int] = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def random(cls, num_pairs: int = 4, scheme: str = "pe", image_size: Union[int, Tuple[int, int]] = 32,
               key_seed: Optional[int] = None, channel_mode: str = "original", block_size: int = 4,
               **kwargs) -> "ScrambleMix":
        """New random key set of ``num_pairs`` key pairs (``key_seed`` makes the keys reproducible).

        ``scheme`` is ``"pe"`` (pixel-based encryption, used by the paper) or ``"le"``
        (block-wise learnable encryption). Remaining keyword arguments go to the constructor.
        """
        keys = random_keys(2 * num_pairs, scheme, image_size, key_seed, channel_mode, block_size)
        return cls(consecutive_pairs(keys), **kwargs)

    @classmethod
    def from_keys(cls, keys: Sequence[object], **kwargs) -> "ScrambleMix":
        """Pair a list of keys as (0, 1), (2, 3), ... like the original code."""
        return cls(consecutive_pairs(list(keys)), **kwargs)

    @classmethod
    def original(cls, dataset: str = "cifar10", channel_mode: str = "original", **kwargs) -> "ScrambleMix":
        """The key set of the original code for ``dataset`` in {cifar10, cifar100, svhn} (32x32 images)."""
        return cls.from_keys(original_pe_keys(dataset, channel_mode), **kwargs)

    # ------------------------------------------------------------------ properties
    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    @property
    def scheme(self) -> str:
        return getattr(self.pairs[0][0], "scheme", "custom")

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        return getattr(self.pairs[0][0], "image_size", None)

    def subset(self, pairs: Iterable[int], **kwargs) -> "ScrambleMix":
        """A ScrambleMix restricted to some key pairs (sharing the keys)."""
        kw = dict(alpha=self.alpha, mix_ratio=self.mix_ratio, seed=self.seed)
        kw.update(kwargs)
        return ScrambleMix([self.pairs[i] for i in pairs], **kw)

    # ------------------------------------------------------------------ randomness
    def _generator(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            entropy = [] if self.seed is None else [int(self.seed)]
            info = torch.utils.data.get_worker_info()
            if info is not None:  # DataLoader worker: independent stream per worker
                entropy.append(int(info.seed) % (2 ** 63))
            self._rng = np.random.default_rng(entropy if entropy else None)
            self._rng_pid = pid
        return self._rng

    def sample_mix_ratio(self, size: Optional[int] = None):
        """Draw ``m ~ Beta(alpha, alpha)`` (or return the fixed ``mix_ratio``)."""
        n = 1 if size is None else size
        if self.mix_ratio is not None:
            m = np.full(n, self.mix_ratio)
        else:
            m = self._generator().beta(self.alpha, self.alpha, size=n)
        return float(m[0]) if size is None else m

    # ------------------------------------------------------------------ core
    def _mix_pair(self, x: torch.Tensor, pair: int, m: torch.Tensor) -> torch.Tensor:
        k1, k2 = self.pairs[pair]
        return ((1.0 - m) * k1(x).float() + m * k2(x).float()) / 255.0

    def _mix(self, x: torch.Tensor, pair_idx: np.ndarray, m: np.ndarray) -> torch.Tensor:
        """x: uint8 [B, 3, H, W]; pair_idx, m: arrays of length B -> float32 [B, 3, H, W] in [0, 1]."""
        m_t = torch.as_tensor(np.asarray(m, dtype=np.float32), device=x.device).view(-1, 1, 1, 1)
        unique = np.unique(pair_idx)
        if unique.size == 1:
            return self._mix_pair(x, int(unique[0]), m_t)
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        for p in unique:
            sel = torch.as_tensor(np.flatnonzero(pair_idx == p), device=x.device)
            out[sel] = self._mix_pair(x[sel], int(p), m_t[sel])
        return out

    def mix(self, img: ImageLike, pair: int = 0, m: Union[float, Sequence[float], None] = None) -> torch.Tensor:
        """``(1 - m) f(x; k_1) + m f(x; k_2)`` for key pair ``pair`` (``m`` sampled if ``None``)."""
        x, was_batched = batched(as_uint8_tensor(img))
        b = x.shape[0]
        m_arr = self.sample_mix_ratio(b) if m is None else np.broadcast_to(np.asarray(m, dtype=np.float64), (b,))
        out = self._mix(x, np.full(b, int(pair)), m_arr)
        return out if was_batched else out[0]

    def sample(self, img: ImageLike, pairs: Optional[Iterable[int]] = None) -> torch.Tensor:
        """One ScrambleMix view per image, with the key pair drawn uniformly from ``pairs`` (default: all)."""
        x, was_batched = batched(as_uint8_tensor(img))
        b = x.shape[0]
        choices = np.arange(self.num_pairs) if pairs is None else np.asarray(list(pairs))
        rng = self._generator()
        pair_idx = choices[rng.integers(0, len(choices), size=b)]
        out = self._mix(x, pair_idx, self.sample_mix_ratio(b))
        return out if was_batched else out[0]

    __call__ = sample

    def views(self, img: ImageLike, pairs: Optional[Iterable[int]] = None) -> torch.Tensor:
        """One view per key pair in ``pairs`` (default: all): ``[D, 3, H, W]`` or ``[B, D, 3, H, W]``."""
        x, was_batched = batched(as_uint8_tensor(img))
        b = x.shape[0]
        pair_list = range(self.num_pairs) if pairs is None else [int(p) for p in pairs]
        out = torch.stack([self._mix(x, np.full(b, p), self.sample_mix_ratio(b)) for p in pair_list], dim=1)
        return out if was_batched else out[0]

    # ------------------------------------------------------------------ (de)serialisation
    def state_dict(self) -> dict:
        """Key set as a dict of tensors and Python scalars (loadable with ``torch.load(weights_only=True)``)."""
        keys: list = []
        index: dict = {}
        pairs = []
        for pair in self.pairs:
            ids = []
            for k in pair:
                if id(k) not in index:
                    index[id(k)] = len(keys)
                    keys.append(k)
                ids.append(index[id(k)])
            pairs.append(ids)
        return {"format": KEYSET_FORMAT, "keys": [k.state_dict() for k in keys], "pairs": pairs,
                "alpha": self.alpha, "mix_ratio": self.mix_ratio}

    @classmethod
    def from_state_dict(cls, state: dict, **kwargs) -> "ScrambleMix":
        if state.get("format") != KEYSET_FORMAT:
            raise ValueError("not a ScrambleMix key set")
        keys = [scrambler_from_state_dict(k) for k in state["keys"]]
        kw = dict(alpha=state["alpha"], mix_ratio=state["mix_ratio"])
        kw.update(kwargs)
        return cls([(keys[i], keys[j]) for i, j in state["pairs"]], **kw)

    def save(self, path: Union[str, Path]) -> None:
        """Save the key set (keep this file secret: it is the edge-side key)."""
        torch.save(self.state_dict(), str(path))

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "ScrambleMix":
        return cls.from_state_dict(torch.load(str(path), map_location="cpu", weights_only=True), **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rng"], state["_rng_pid"] = None, None
        return state

    def __repr__(self) -> str:
        size = self.image_size
        ratio = f"mix_ratio={self.mix_ratio}" if self.mix_ratio is not None else f"alpha={self.alpha}"
        return (f"ScrambleMix(num_pairs={self.num_pairs}, scheme={self.scheme!r}, {ratio}"
                + (f", image_size={size}" if size else "") + ")")


class ScrambleMixViews:
    """Transform returning one ScrambleMix view per key pair, stacked as ``[D, 3, H, W]``.

    These are the ``D`` training views of slide 29 (key pairs ``0 .. D-1``); a DataLoader then
    yields ``[B, D, 3, H, W]`` batches for :func:`scramblemix.scramblemix_loss`.
    """

    def __init__(self, scramblemix: ScrambleMix, num_views: Optional[int] = None) -> None:
        num_views = scramblemix.num_pairs if num_views is None else int(num_views)
        if not 1 <= num_views <= scramblemix.num_pairs:
            raise ValueError(f"num_views must be in 1..{scramblemix.num_pairs}")
        self.scramblemix = scramblemix
        self.num_views = num_views

    def __call__(self, img: ImageLike) -> torch.Tensor:
        return self.scramblemix.views(img, range(self.num_views))

    def __repr__(self) -> str:
        return f"ScrambleMixViews({self.scramblemix!r}, num_views={self.num_views})"


class EvalViews:
    """Test-time transform used by the original code; returns ``(single, tta)``.

    * ``single`` -- one ScrambleMix view whose key pair is drawn from the first ``num_train_views``
      pairs (the ``T = 1`` setting of slide 39);
    * ``tta`` -- ``[T, 3, H, W]`` views for key pairs ``0 .. T-1`` (ScrambleMix TTA, slides 37 and 40).
    """

    def __init__(self, scramblemix: ScrambleMix, num_train_views: int, num_tta_views: int) -> None:
        if not 1 <= num_train_views <= scramblemix.num_pairs or not 1 <= num_tta_views <= scramblemix.num_pairs:
            raise ValueError(f"view counts must be in 1..{scramblemix.num_pairs}")
        self.scramblemix = scramblemix
        self.num_train_views = int(num_train_views)
        self.num_tta_views = int(num_tta_views)

    def __call__(self, img: ImageLike) -> Tuple[torch.Tensor, torch.Tensor]:
        x = as_uint8_tensor(img)
        single = self.scramblemix.sample(x, range(self.num_train_views))
        tta = self.scramblemix.views(x, range(self.num_tta_views))
        return single, tta

    def __repr__(self) -> str:
        return (f"EvalViews({self.scramblemix!r}, num_train_views={self.num_train_views}, "
                f"num_tta_views={self.num_tta_views})")
