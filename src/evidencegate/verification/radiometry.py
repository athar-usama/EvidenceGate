"""Colour/contrast plausibility scoring, the third independent evidence signal.

Neither the proposer nor the segmentation cross-check looks at colour directly
in Lab space; this catches cases where geometry and grounding agree but the
region simply isn't the right colour for the claimed lesion family (e.g. a
vessel crossing or an artifact that both other signals missed).
"""

from __future__ import annotations

import cv2
import numpy as np

from evidencegate.geometry import Box


def _lab_stats(bgr_region: np.ndarray) -> tuple[float, float, float]:
    lab = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(lab[..., 0].mean()), float(lab[..., 1].mean()), float(lab[..., 2].mean())


def plausibility_score(bgr: np.ndarray, claim_box: Box, family: str, background_margin: float = 1.5) -> float:
    """Returns a score in [0, 1]; higher means the region's colour matches the claimed family."""
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = claim_box.clip(width, height).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = bgr[y1:y2, x1:x2]

    outer = claim_box.expand(background_margin, width, height)
    ox1, oy1, ox2, oy2 = outer.as_int_tuple()
    surround = bgr[oy1:oy2, ox1:ox2]
    inset_y1, inset_x1 = y1 - oy1, x1 - ox1
    background_selector = np.ones(surround.shape[:2], dtype=bool)
    background_selector[inset_y1 : inset_y1 + (y2 - y1), inset_x1 : inset_x1 + (x2 - x1)] = False
    background_pixels = surround[background_selector]
    if background_pixels.size == 0:
        return 0.0
    background_pixels = background_pixels.reshape(-1, 1, 3).astype(np.uint8)

    region_l, region_a, region_b = _lab_stats(region)
    back_l, back_a, back_b = _lab_stats(background_pixels)

    if family == "red":
        # darker than background (lower L) and shifted toward red on the a* axis
        darkness = np.clip((back_l - region_l) / 20.0, 0.0, 1.0)
        redness = np.clip((region_a - back_a) / 12.0, 0.0, 1.0)
        return float(0.5 * darkness + 0.5 * redness)

    if family == "bright":
        # brighter than background (higher L) and low blue-yellow deficit (b* shifted positive/yellow)
        brightness = np.clip((region_l - back_l) / 20.0, 0.0, 1.0)
        yellowness = np.clip((region_b - back_b) / 12.0, 0.0, 1.0)
        return float(0.5 * brightness + 0.5 * yellowness)

    return 0.0
