"""Shared classical preprocessing for the candidate proposers.

Every function here is deterministic, closed-form image processing (no learned
weights) so that the proposers stay a fully independent, inspectable evidence
source relative to the VLM grounding stage.
"""

from __future__ import annotations

import cv2
import numpy as np

from evidencegate.geometry import Box


def retinal_field_mask(bgr: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Binary mask of the circular fundus field, excluding the black surround."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def enhanced_green_channel(bgr: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced green channel, the standard contrast channel for fundus lesions."""
    green = bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    enhanced[field_mask == 0] = int(np.median(enhanced[field_mask > 0])) if (field_mask > 0).any() else 0
    return enhanced


def locate_optic_disc(bgr: np.ndarray, field_mask: np.ndarray) -> Box | None:
    """Localize the optic disc as the largest very-bright compact blob.

    The disc is the brightest structure in a healthy or DR fundus image and would
    otherwise dominate the bright-lesion (exudate) proposer as a false candidate.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    masked = np.where(field_mask > 0, gray, 0)
    bright_thresh = np.percentile(masked[field_mask > 0], 99.0) if (field_mask > 0).any() else 255
    bright_mask = ((masked >= bright_thresh) & (field_mask > 0)).astype(np.uint8) * 255
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x, y, w, h, _ = stats[best]
    return Box(float(x), float(y), float(x + w), float(y + h))
