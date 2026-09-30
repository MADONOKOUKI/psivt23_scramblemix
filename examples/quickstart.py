"""ScrambleMix quick start: scramble a photo with two keys, mix the two scrambles, measure LPIPS.

    python examples/quickstart.py            # writes assets/quickstart.png

Runs on a CPU in a few seconds and needs no dataset. The first run downloads the AlexNet weights
used by LPIPS (~233 MB, via torchvision).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

from scramblemix import ScrambleMix, lpips_distance  # noqa: E402
from scramblemix._image import as_float_tensor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "quickstart.png"
MIX_RATIOS = (0.1, 0.3, 0.5)  # the mixing ratios shown on slide 42 of the paper's slides


def scramble_panels(photo: Image.Image, size: int):
    """Return [(title, image [3, H, W] in [0, 1])] for one image size."""
    img = photo.resize((size, size), Image.BICUBIC)
    # a secret key pair (k1, k2) of pixel-based encryption keys for size x size images
    sm = ScrambleMix.random(num_pairs=1, scheme="pe", image_size=size, key_seed=0)
    k1, k2 = sm.pairs[0]
    panels = [("original x", as_float_tensor(img)),
              ("f(x; k1)", k1(img).float() / 255),
              ("f(x; k2)", k2(img).float() / 255)]
    panels += [(f"ScrambleMix, m = {m}", sm.mix(img, pair=0, m=m)) for m in MIX_RATIOS]
    return panels


def main() -> None:
    torch.manual_seed(0)
    photo = Image.fromarray(data.astronaut())  # 512x512 RGB image bundled with scikit-image
    sizes = (32, 128)
    rows = [scramble_panels(photo, s) for s in sizes]

    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(2.3 * len(rows[0]), 2.75 * len(rows)))
    print(f"LPIPS (AlexNet) to the original image, higher = better visual information hiding")
    for r, (size, panels) in enumerate(zip(sizes, rows)):
        original = panels[0][1]
        scrambled = torch.stack([p[1] for p in panels[1:]])
        scores = lpips_distance(original.expand_as(scrambled), scrambled)
        print(f"  {size}x{size}: " + ", ".join(f"{t}: {s:.3f}" for (t, _), s in zip(panels[1:], scores.tolist())))
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{size} x {size}" + (" (paper)" if size == 32 else ""), fontsize=10)
            else:
                ax.set_xlabel(f"LPIPS = {scores[c - 1]:.3f}", fontsize=10)
    fig.suptitle("ScrambleMix:  (1 - m) f(x; k1) + m f(x; k2)   (pixel-based encryption keys, LPIPS vs. original)",
                 fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
