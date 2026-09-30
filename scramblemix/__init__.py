"""ScrambleMix: A Privacy-Preserving Image Processing for Edge-Cloud Machine Learning (PSIVT 2023).

Official PyTorch implementation. Main entry points:

* :class:`ScrambleMix` -- the transform ``(1 - m) f(x; k1) + m f(x; k2)``, ``m ~ Beta(alpha, alpha)``
* :class:`PixelEncryption`, :class:`LearnableEncryption` -- the scrambling functions ``f(x; k)``
* :func:`scramblemix_loss`, :func:`self_teaching_loss` -- cross-entropy + self-teaching loss
* :func:`predict_tta` -- multi-key test-time augmentation
* :func:`lpips_distance` -- LPIPS evaluation of visual information hiding
"""
from .keys import load_le_keys, original_pe_keys, random_keys
from .losses import scramblemix_loss, self_teaching_loss
from .metrics import lpips_distance, lpips_model
from .scrambling import LearnableEncryption, PixelEncryption
from .transform import EvalViews, ScrambleMix, ScrambleMixViews
from .tta import average_predictions, predict_tta

__version__ = "1.0.0"

__all__ = [
    "ScrambleMix", "ScrambleMixViews", "EvalViews",
    "PixelEncryption", "LearnableEncryption",
    "original_pe_keys", "load_le_keys", "random_keys",
    "scramblemix_loss", "self_teaching_loss",
    "predict_tta", "average_predictions",
    "lpips_distance", "lpips_model",
]
