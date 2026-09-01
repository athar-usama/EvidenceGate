"""Classical candidate proposer for red lesions (microaneurysms, haemorrhages).

Recall-oriented on purpose: this stage only has to surface plausible regions,
not decide anything. Precision is the job of the consensus + gating stages
downstream. Method: black-hat morphology (dark blobs on a brighter local
background) with a shape filter to reject elongated vessel fragments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from evidencegate.geometry import Box
from evidencegate.proposers.preprocess import enhanced_green_channel, retinal_field_mask


@dataclass(frozen=True)
class Candidate:
    box: Box
    circularity: float
    contrast: float
    lesion_family: str


def propose_red_lesions(
    bgr: np.ndarray,
    min_area: int = 12,
    max_area: int = 6000,
    min_circularity: float = 0.35,
    structuring_element_size: int = 9,
    max_candidates: int = 20,
    denoise_ksize: int = 9,
) -> list[Candidate]:
    """`denoise_ksize` matters a lot in practice: without it, pixel-scale sensor/JPEG noise
    produces thousands of spuriously "high-contrast" black-hat responses that rank far above
    genuine (subtler) microaneurysms once sorted by contrast for the `max_candidates` cutoff —
    a median blur before black-hat suppresses that noise floor without erasing lesion-scale blobs.
    """
    field_mask = retinal_field_mask(bgr)
    green = enhanced_green_channel(bgr, field_mask)
    if denoise_ksize:
        green = cv2.medianBlur(green, denoise_ksize)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structuring_element_size, structuring_element_size))
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    blackhat[field_mask == 0] = 0

    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates: list[Candidate] = []
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = stats[label, cv2.CC_STAT_LEFT : cv2.CC_STAT_HEIGHT + 1]
        # Crop to the component's own bounding box before masking/contouring — comparing against
        # the full-resolution label map per component (a ~12MP image here) is the difference
        # between this running in milliseconds and minutes once hundreds of components pass the
        # area filter.
        local_labels = labels[y : y + h, x : x + w]
        component_mask = (local_labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = float(4 * np.pi * area / (perimeter**2))
        if circularity < min_circularity:
            continue  # rejects elongated vessel fragments

        local_background = np.median(green[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3])
        local_signal = np.median(green[y : y + h, x : x + w])
        contrast = float(local_background) - float(local_signal)

        candidates.append(
            Candidate(
                box=Box(float(x), float(y), float(x + w), float(y + h)),
                circularity=circularity,
                contrast=contrast,
                lesion_family="red",
            )
        )
    candidates.sort(key=lambda c: c.contrast, reverse=True)
    return candidates[:max_candidates]
