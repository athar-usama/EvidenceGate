"""Loads the reorganized IDRiD segmentation data (see scripts/download_data.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

FAMILY_TO_CATEGORIES = {"red": ["MA", "HE"], "bright": ["EX", "SE"]}


@dataclass(frozen=True)
class FundusImage:
    image_id: str
    split: str
    image_path: Path

    def load_bgr(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(self.image_path)
        return image


def list_images(split: str) -> list[FundusImage]:
    directory = RAW_DIR / "images" / split
    return [
        FundusImage(image_id=path.stem, split=split, image_path=path)
        for path in sorted(directory.glob("*.jpg"))
    ]


def _mask_path(category: str, split: str, image_id: str) -> Path:
    return RAW_DIR / "masks" / category / split / f"{image_id}_{category}.tif"


def load_family_mask(image_id: str, split: str, family: str, shape: tuple[int, int]) -> np.ndarray:
    """Union mask (H, W) uint8 {0, 1} across the categories that make up a lesion family."""
    union = np.zeros(shape, dtype=np.uint8)
    for category in FAMILY_TO_CATEGORIES[family]:
        path = _mask_path(category, split, image_id)
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        union |= (mask > 0).astype(np.uint8)
    return union


def load_optic_disc_mask(image_id: str, split: str, shape: tuple[int, int]) -> np.ndarray:
    path = _mask_path("OD", split, image_id)
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)
