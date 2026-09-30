"""LPIPS evaluation of visual information hiding.

The paper quantifies how well scrambled images hide the visual content of the original with
the Learned Perceptual Image Patch Similarity (LPIPS, Zhang et al., CVPR 2018) between the
original and the scrambled image: the *larger* the LPIPS distance, the less the scrambled image
reveals. The original code uses ``lpips.LPIPS(net='alex')`` on 32x32 test images.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import torch

from ._image import ImageLike, as_float_tensor

_MODELS: Dict[Tuple[str, str, bool], torch.nn.Module] = {}


def lpips_model(net: str = "alex", device: Union[str, torch.device] = "cpu",
                pretrained_backbone: bool = True) -> torch.nn.Module:
    """Cached ``lpips.LPIPS`` model. The backbone weights are downloaded by torchvision on first use;
    ``pretrained_backbone=False`` gives a random backbone (no download, for tests only)."""
    key = (net, str(device), pretrained_backbone)
    if key not in _MODELS:
        import lpips  # imported lazily: only needed for the evaluation

        with warnings.catch_warnings():  # torchvision's 'pretrained' deprecation warnings inside lpips
            warnings.simplefilter("ignore", UserWarning)
            model = lpips.LPIPS(net=net, pnet_rand=not pretrained_backbone, verbose=False)
        _MODELS[key] = model.to(device).eval()
    return _MODELS[key]


@torch.no_grad()
def lpips_distance(original: ImageLike, scrambled: ImageLike, net: str = "alex",
                   model: Optional[torch.nn.Module] = None, device: Union[str, torch.device, None] = None,
                   batch_size: int = 256) -> torch.Tensor:
    """Per-image LPIPS distance between ``original`` and ``scrambled`` (higher = better hiding).

    Both inputs are images in the :class:`~scramblemix.ScrambleMix` output convention
    (``[B, 3, H, W]`` or ``[3, H, W]`` float tensors in ``[0, 1]``; ``uint8`` tensors, PIL images
    and numpy arrays are converted). Returns a float tensor ``[B]`` on the CPU.
    """
    x = as_float_tensor(original)
    y = as_float_tensor(scrambled)
    if x.ndim == 3:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if model is None:
        device = device or "cpu"
        model = lpips_model(net, device)
    else:
        device = device or next(model.parameters()).device
    out = []
    for i in range(0, x.shape[0], batch_size):
        a = x[i:i + batch_size].to(device)
        b = y[i:i + batch_size].to(device)
        out.append(model(a, b, normalize=True).flatten().float().cpu())  # normalize: [0, 1] -> [-1, 1]
    return torch.cat(out)
