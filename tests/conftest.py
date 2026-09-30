import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
ORIGINAL_CODE = ARCHIVE / "scripts" / "scramblemix"


def load_archived_module(relpath: str, name: str):
    """Import a file of the original research code (archive/) by path; skip if archive/ is absent."""
    path = ARCHIVE / relpath
    if not path.exists():
        pytest.skip("archive/ (original code) not available")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def original_augmix_module(monkeypatch):
    """The original ``cifar10.py`` (AugMixDataset + hard-coded keys), imported with a clean argv."""
    if not (ORIGINAL_CODE / "cifar10.py").exists():
        pytest.skip("archive/ (original code) not available")
    names = ["cifar10", "parameter", "pixel_based_encryption", "Blockwise_scramble_LE", "learnable_encryption_augmix"]
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    monkeypatch.setattr(sys, "argv", ["train", "--dataset", "cifar10", "--num_of_TTA", "4"])
    monkeypatch.syspath_prepend(str(ORIGINAL_CODE))
    import cifar10  # noqa: F401  (the archived module)

    module = sys.modules["cifar10"]
    yield module
    for n in names:
        sys.modules.pop(n, None)
    sys.modules.update(saved)
