"""Datasets used in the paper (CIFAR-10, CIFAR-100, SVHN) plus a synthetic ``fake`` dataset for smoke tests."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "svhn": 10, "fake": 10}


def train_augmentation() -> transforms.Compose:
    """PIL augmentation applied before scrambling, as in the original ``dataloader.py``."""
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])


def build_datasets(name: str, root: str = "./data", train_transform: Optional[Callable] = None,
                   test_transform: Optional[Callable] = None, download: bool = True,
                   fake_train_size: int = 512, fake_test_size: int = 256) -> Tuple[Dataset, Dataset, int]:
    """Return ``(train_set, test_set, num_classes)``; images are PIL images passed to the transforms.

    ``name="fake"`` uses ``torchvision.datasets.FakeData`` (random 32x32 images, 10 classes, no download).
    """
    if name == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=download, transform=train_transform)
        test = datasets.CIFAR10(root, train=False, download=download, transform=test_transform)
    elif name == "cifar100":
        train = datasets.CIFAR100(root, train=True, download=download, transform=train_transform)
        test = datasets.CIFAR100(root, train=False, download=download, transform=test_transform)
    elif name == "svhn":
        train = datasets.SVHN(root, split="train", download=download, transform=train_transform)
        test = datasets.SVHN(root, split="test", download=download, transform=test_transform)
    elif name == "fake":
        train = datasets.FakeData(fake_train_size, (3, 32, 32), 10, transform=train_transform, random_offset=0)
        test = datasets.FakeData(fake_test_size, (3, 32, 32), 10, transform=test_transform, random_offset=10 ** 6)
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(NUM_CLASSES)}")
    return train, test, NUM_CLASSES[name]
