"""Classical candidate proposer for bright lesions (hard & soft exudates).

Same recall-oriented philosophy as the red-lesion proposer, mirrored for bright
blobs: top-hat morphology (bright spots on a darker local background), with the
optic disc explicitly excluded since it is the single brightest structure in
almost every fundus image and would otherwise swamp the proposer.
"""

from __future__ import annotations

import cv2
import numpy as np

from evidencegate.geometry import Box, iou
from evidencegate.proposers.preprocess import enhanced_green_channel, locate_optic_disc, retinal_field_mask
from evidencegate.proposers.red_lesion import Candidate


def propose_bright_lesions(
    bgr: np.ndarray,
    min_area: int = 20,
    max_area: int = 20000,
    structuring_element_size: int = 15,
    optic_disc_iou_guard: float = 0.05,
    max_candidates: int = 20,
    denoise_ksize: int = 9,
) -> list[Candidate]:
    """See red_lesion.py's `denoise_ksize` docstring, the same noise-floor problem shows up here."""
    field_mask = retinal_field_mask(bgr)
    green = enhanced_green_channel(bgr, field_mask)
    if denoise_ksize:
        green = cv2.medianBlur(green, denoise_ksize)
    optic_disc = locate_optic_disc(bgr, field_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_element_size, structuring_element_size))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    tophat[field_mask == 0] = 0

    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[Candidate] = []
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = stats[label, 0:4]
        box = Box(float(x), float(y), float(x + w), float(y + h))

        if optic_disc is not None and iou(box, optic_disc) > optic_disc_iou_guard:
            continue

        # See red_lesion.py: crop to the component's bounding box before masking/contouring.
        local_labels = labels[y : y + h, x : x + w]
        component_mask = (local_labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        perimeter = cv2.arcLength(contour, True) if contour is not None else 0.0
        circularity = float(4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

        local_background = np.median(green[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3])
        local_signal = np.median(green[y : y + h, x : x + w])
        contrast = float(local_signal) - float(local_background)

        candidates.append(
            Candidate(box=box, circularity=circularity, contrast=contrast, lesion_family="bright")
        )
    candidates.sort(key=lambda c: c.contrast, reverse=True)
    return candidates[:max_candidates]
