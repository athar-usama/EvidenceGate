"""Ground-truth correctness criterion shared by calibration and evaluation scripts."""

from __future__ import annotations

import cv2
import numpy as np

from evidencegate.geometry import Box


def is_correct_claim(box: Box, family_mask: np.ndarray, min_overlap_frac: float = 0.15) -> bool:
    """A claim is correct if a meaningful fraction of its box actually covers true lesion pixels."""
    height, width = family_mask.shape
    x1, y1, x2, y2 = box.clip(width, height).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return False
    region = family_mask[y1:y2, x1:x2]
    box_area = region.size
    if box_area == 0:
        return False
    overlap_frac = float(region.sum()) / box_area
    return overlap_frac >= min_overlap_frac


def recall_against_instances(boxes: list[Box], family_mask: np.ndarray) -> tuple[int, int]:
    """Returns (n_lesion_instances_hit, n_lesion_instances_total) via connected components of the mask.

    A lesion instance counts as "hit" if any given box overlaps at least one of its pixels —
    a deliberately generous criterion, since the point here is recall of the *gate*, not of box tightness.
    """
    n_labels, labels = cv2.connectedComponents(family_mask.astype(np.uint8), connectivity=8)
    n_instances = n_labels - 1
    if n_instances == 0:
        return 0, 0

    height, width = family_mask.shape
    hit_label_ids: set[int] = set()
    for box in boxes:
        x1, y1, x2, y2 = box.clip(width, height).as_int_tuple()
        if x2 <= x1 or y2 <= y1:
            continue
        region_labels = labels[y1:y2, x1:x2]
        hit_label_ids.update(int(v) for v in np.unique(region_labels) if v != 0)

    return len(hit_label_ids), n_instances
