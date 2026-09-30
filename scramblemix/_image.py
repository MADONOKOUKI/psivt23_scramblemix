"""Conversion helpers shared by the scramblers and the ScrambleMix transform."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]


def as_uint8_tensor(img: ImageLike) -> torch.Tensor:
    """Convert an image to a ``uint8`` tensor of shape ``[3, H, W]`` or ``[B, 3, H, W]``.

    Accepted inputs:

    * ``PIL.Image`` (converted to RGB),
    * ``numpy`` arrays ``[H, W, 3]`` or ``[B, H, W, 3]``,
    * ``torch`` tensors ``[3, H, W]`` or ``[B, 3, H, W]``.

    ``uint8`` data is used as is; floating-point data is assumed to be in ``[0, 1]``
    (the ``ToTensor`` convention) and is rounded to the nearest 8-bit value.
    """
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
    if isinstance(img, np.ndarray):
        if img.ndim not in (3, 4) or img.shape[-1] != 3:
            raise ValueError(f"expected a numpy array of shape [H, W, 3] or [B, H, W, 3], got {img.shape}")
        arr = np.ascontiguousarray(img)
        if not arr.flags.writeable:  # e.g. np.asarray(PIL image); torch warns on read-only arrays
            arr = arr.copy()
        t = torch.from_numpy(arr).movedim(-1, -3)
    elif isinstance(img, torch.Tensor):
        if img.ndim not in (3, 4) or img.shape[-3] != 3:
            raise ValueError(f"expected a tensor of shape [3, H, W] or [B, 3, H, W], got {tuple(img.shape)}")
        t = img
    else:
        raise TypeError(f"unsupported image type: {type(img).__name__}")
    if t.dtype == torch.uint8:
        return t.contiguous()
    if t.is_floating_point():
        return (t.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
    return t.clamp(0, 255).to(torch.uint8).contiguous()


def as_float_tensor(img: ImageLike) -> torch.Tensor:
    """Like :func:`as_uint8_tensor` but returns ``float32`` in ``[0, 1]``; float tensors are kept unquantised."""
    if isinstance(img, torch.Tensor) and img.is_floating_point():
        if img.ndim not in (3, 4) or img.shape[-3] != 3:
            raise ValueError(f"expected a tensor of shape [3, H, W] or [B, 3, H, W], got {tuple(img.shape)}")
        return img.float()
    return as_uint8_tensor(img).float() / 255.0


def batched(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(x[None] if x is a single image else x, was_batched)``."""
    if x.ndim == 3:
        return x.unsqueeze(0), False
    return x, True


def pair(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)
