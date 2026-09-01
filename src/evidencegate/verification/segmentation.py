"""Independent segmentation cross-check for a consensus claim.

Deliberately a *different* classical technique from the candidate proposers
(GrabCut graph-cut segmentation vs. top-hat/black-hat morphology) so that this
verification signal doesn't just re-derive the same evidence the proposer
already used — it's the "invoke a specialized segmentation tool" step from the
source post, and it needs to be able to disagree with the grounding stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from evidencegate.geometry import Box, iou


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray  # binary mask, same size as the padded crop
    crop_box: Box  # padded crop region in original-image coordinates
    segmented_box: Box | None  # tight bounding box of the segmented mask, in original-image coords
    fill_ratio: float  # segmented area / claim-box area — expected near 1 for a well-localized claim
    foreground_fraction: float  # share of the whole crop marked foreground — near 1 means GrabCut gave up and marked everything, which is not corroboration
    iou_with_claim: float

    @property
    def agreement(self) -> float:
        """Coverage of the claimed pixels specifically — does GrabCut, run independently and
        knowing nothing about the grounder's answer, also consider this exact region foreground?
        A degenerate "mark everything foreground" GrabCut run would trivially max this out, so it
        is combined with `foreground_fraction` downstream only as a sanity/debug signal, not
        folded into the score itself: empirically, penalizing high foreground_fraction directly
        punished large-but-genuine lesions (soft exudates in particular) more than it caught
        degenerate segmentations, which made it net-harmful rather than net-helpful."""
        return float(min(1.0, self.fill_ratio))


def segment_claim(bgr: np.ndarray, claim_box: Box, padding_frac: float = 1.2, min_crop_size: int = 64) -> SegmentationResult:
    height, width = bgr.shape[:2]
    padded = claim_box.expand(padding_frac, width, height)
    if padded.x2 - padded.x1 < min_crop_size or padded.y2 - padded.y1 < min_crop_size:
        cx, cy = claim_box.center()
        half = min_crop_size / 2
        padded = Box(cx - half, cy - half, cx + half, cy + half)
    crop_box = padded.clip(width, height)
    cx1, cy1, cx2, cy2 = crop_box.as_int_tuple()
    crop = bgr[cy1:cy2, cx1:cx2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return SegmentationResult(
            mask=np.zeros((1, 1), np.uint8), crop_box=crop_box, segmented_box=None,
            fill_ratio=0.0, foreground_fraction=0.0, iou_with_claim=0.0,
        )

    grabcut_mask = np.zeros(crop.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect_x1 = int(round(claim_box.x1 - cx1))
    rect_y1 = int(round(claim_box.y1 - cy1))
    rect_x2 = int(round(claim_box.x2 - cx1))
    rect_y2 = int(round(claim_box.y2 - cy1))
    rect = (
        max(0, rect_x1),
        max(0, rect_y1),
        max(1, min(rect_x2, crop.shape[1] - 1) - max(0, rect_x1)),
        max(1, min(rect_y2, crop.shape[0] - 1) - max(0, rect_y1)),
    )

    try:
        cv2.grabCut(crop, grabcut_mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
        binary_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        binary_mask = np.zeros(crop.shape[:2], np.uint8)
        binary_mask[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]] = 1

    foreground_fraction = float(binary_mask.mean())
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return SegmentationResult(
            mask=binary_mask, crop_box=crop_box, segmented_box=None,
            fill_ratio=0.0, foreground_fraction=foreground_fraction, iou_with_claim=0.0,
        )

    segmented_box = Box(
        float(xs.min() + cx1), float(ys.min() + cy1), float(xs.max() + 1 + cx1), float(ys.max() + 1 + cy1)
    )
    # Coverage of the *claimed* pixels specifically, not just the segmented bounding box overall —
    # matters at few-pixel lesion scale, where GrabCut's inferred box rarely matches exactly.
    cbx1, cby1, cbx2, cby2 = claim_box.clip(width, height).as_int_tuple()
    local_x1, local_y1 = max(0, cbx1 - cx1), max(0, cby1 - cy1)
    local_x2, local_y2 = min(binary_mask.shape[1], cbx2 - cx1), min(binary_mask.shape[0], cby2 - cy1)
    claim_region_mask = binary_mask[local_y1:local_y2, local_x1:local_x2]
    fill_ratio = float(claim_region_mask.sum()) / max(1.0, claim_region_mask.size)
    overlap = iou(segmented_box, claim_box)

    return SegmentationResult(
        mask=binary_mask, crop_box=crop_box, segmented_box=segmented_box,
        fill_ratio=fill_ratio, foreground_fraction=foreground_fraction, iou_with_claim=overlap,
    )
