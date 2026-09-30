"""Training objective of ScrambleMix: cross-entropy + self-teaching loss (slides 30-36).

For an image ``x_i`` with label ``y_i`` and its ``D`` ScrambleMix views, the classifier
predicts posteriors ``y^_{i,1..D}``. The loss is

    L    = L_CE + lambda * L_ST                                             (slide 30)
    L_CE = 1/(BD) sum_i sum_d CE(y^_{i,d}, y_i)                              (slide 31)
    y-_i = StopGrad( 1/D sum_d y^_{i,d} )                                    (slide 35)
    L_ST = 1/(BD) sum_i sum_d KL(y^_{i,d} || y-_i)                           (slide 36)

The self-teaching loss pulls the posteriors of the differently keyed views of the same image
towards their average. In the original code (``archive/scripts/scramblemix/trainer.py``,
option ``--js_divergence_regularization``) it is written as the generalised Jensen-Shannon
divergence ``1/D sum_d KL(p_d || mean_d p_d)`` *without* the stop-gradient and with
``lambda = 1``. Both forms have the same value and the same gradient with respect to the
logits (the extra gradient through the average is ``1/D`` for every class, which the softmax
Jacobian removes), so ``stop_grad`` only changes the autograd graph; this is checked in
``tests/test_losses.py``.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn.functional as F

Logits = Union[torch.Tensor, Sequence[torch.Tensor]]


def _stack(logits: Logits) -> torch.Tensor:
    z = logits if isinstance(logits, torch.Tensor) else torch.stack(list(logits), dim=0)
    if z.ndim != 3:
        raise ValueError(f"expected logits of shape [D, B, K] (or a list of D [B, K] tensors), got {tuple(z.shape)}")
    return z


def self_teaching_loss(logits: Logits, stop_grad: bool = True) -> torch.Tensor:
    """Self-teaching loss ``L_ST`` (slides 32-36).

    Args:
        logits: ``[D, B, K]`` tensor, or a list of ``D`` tensors ``[B, K]``, holding the logits of
            the ``D`` ScrambleMix views of each image.
        stop_grad: detach the average posterior (slide 35). The original code does not detach it;
            the gradient w.r.t. the logits is identical either way.

    Returns:
        Scalar ``1/(BD) sum_i sum_d KL(y^_{i,d} || y-_i)``; zero when ``D == 1``.
    """
    z = _stack(logits)
    log_p = F.log_softmax(z, dim=-1)
    # log of the average posterior, computed in log space (no clamp/underflow issues)
    log_mean = torch.logsumexp(log_p, dim=0) - math.log(z.shape[0])
    if stop_grad:
        log_mean = log_mean.detach()
    return (log_p.exp() * (log_p - log_mean)).sum(dim=-1).mean()


def scramblemix_loss(logits: Logits, target: torch.Tensor, lam: float = 1.0, stop_grad: bool = True) -> torch.Tensor:
    """``L = L_CE + lam * L_ST`` (slide 30) for the logits ``[D, B, K]`` of ``D`` views and labels ``[B]``.

    ``lam = 1`` is the weight used by the original code; ``lam = 0`` trains with cross-entropy only.
    """
    z = _stack(logits)
    d, b, k = z.shape
    loss = F.cross_entropy(z.reshape(d * b, k), target.repeat(d))
    if lam:
        loss = loss + lam * self_teaching_loss(z, stop_grad=stop_grad)
    return loss
