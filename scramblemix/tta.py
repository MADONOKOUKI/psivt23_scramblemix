"""ScrambleMix test-time augmentation (slide 37).

The edge device scrambles the input with ``T`` key pairs of the training key set and the
cloud model averages the ``T`` predictions: ``y^_j = 1/T sum_t y^_{j,t}``.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ._image import ImageLike, as_uint8_tensor
from .transform import ScrambleMix

REDUCTIONS = ("logits", "probs")


@torch.no_grad()
def average_predictions(model: torch.nn.Module, views: Union[torch.Tensor, Sequence[torch.Tensor]],
                        reduction: str = "logits") -> torch.Tensor:
    """Average the model's predictions over ``T`` views.

    Args:
        model: classifier returning logits ``[B, K]`` (call ``model.eval()`` first).
        views: ``[B, T, 3, H, W]`` tensor or list of ``T`` tensors ``[B, 3, H, W]``.
        reduction: ``"logits"`` averages logits, as the original code does
            (``archive/scripts/scramblemix/trainer.py``); ``"probs"`` averages softmax
            posteriors, as written on slide 37. Both give ``[B, K]``; take ``argmax(1)`` for labels.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    view_list = list(views.unbind(1)) if isinstance(views, torch.Tensor) else list(views)
    total = None
    for v in view_list:
        out = model(v)
        out = out if reduction == "logits" else F.softmax(out, dim=1)
        total = out if total is None else total + out
    return total / len(view_list)


@torch.no_grad()
def predict_tta(model: torch.nn.Module, images: ImageLike, scramblemix: ScrambleMix,
                num_keys: Optional[int] = None, pairs: Optional[Iterable[int]] = None,
                reduction: str = "logits", device: Union[str, torch.device, None] = None) -> torch.Tensor:
    """Scramble ``images`` with ``T`` key pairs and return the averaged predictions ``[B, K]``.

    Args:
        model: classifier trained on ScrambleMix views (call ``model.eval()`` first).
        images: clean images (``[B, 3, H, W]`` / ``[3, H, W]`` tensor in ``[0, 1]`` or ``uint8``,
            PIL image or numpy array) -- scrambling happens here, on the "edge side".
        scramblemix: the training key set.
        num_keys: ``T``, uses key pairs ``0 .. T-1`` (default: all pairs). Ignored if ``pairs`` is given.
        pairs: explicit key-pair indices.
        reduction: ``"logits"`` (original code) or ``"probs"`` (slide 37).
        device: where to run the model (default: the model's device).
    """
    if device is None:
        device = next(model.parameters()).device
    x = as_uint8_tensor(images)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if pairs is None:
        pairs = range(scramblemix.num_pairs if num_keys is None else num_keys)
    views = scramblemix.views(x, pairs).to(device)
    return average_predictions(model, views, reduction)
