"""Composes the flagship 'walk one finding through the gate' evidence card for a single claim.

Left panel: the region itself, with every individual consensus-run guess drawn
as a faint translucent box (a spatial agreement 'glow': tight overlapping
guesses read as a bright core, scattered guesses read as a diffuse haze), the
final consensus box on top, and the independent segmentation boundary overlaid.
Right panel: the evidence breakdown as a trust ring plus a compact spec sheet,
deliberately not a bar chart.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evidencegate.geometry import Box
from evidencegate.pipeline import Claim, TIER_ABSTAINED, TIER_FLAGGED, TIER_VERIFIED
from evidencegate.report.fonts import font

TIER_COLORS = {
    TIER_VERIFIED: (26, 152, 80),
    TIER_FLAGGED: (217, 164, 6),
    TIER_ABSTAINED: (192, 57, 43),
}
SEGMENTATION_COLOR = (0, 200, 255)
CARD_BG = (18, 20, 24)
PANEL_BG = (28, 31, 38)
TEXT_PRIMARY = (235, 236, 240)
TEXT_SECONDARY = (150, 155, 165)


def _draw_ring(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, frac: float, color: tuple[int, int, int]) -> None:
    draw.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=(60, 63, 70), width=8)
    if frac > 0.001:
        draw.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * min(1.0, frac), fill=color, width=8)


def render_evidence_card(
    bgr: np.ndarray,
    claim: Claim,
    out_path: str,
    crop_padding: float = 2.5,
    panel_width: int = 380,
    min_crop_size: int = 160,
) -> None:
    height, width = bgr.shape[:2]
    region = claim.report_box.expand(crop_padding, width, height)
    if region.x2 - region.x1 < min_crop_size or region.y2 - region.y1 < min_crop_size:
        cx, cy = region.center()
        half = min_crop_size / 2
        region = Box(cx - half, cy - half, cx + half, cy + half).clip(width, height)

    cx1, cy1, cx2, cy2 = region.as_int_tuple()
    crop_bgr = bgr[cy1:cy2, cx1:cx2].copy()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    left = Image.fromarray(crop_rgb).convert("RGBA")
    side = max(left.width, left.height, 340)
    left = left.resize((side, side)) if left.width != left.height else left
    scale_x, scale_y = left.width / crop_rgb.shape[1], left.height / crop_rgb.shape[0]

    def to_local(box: Box) -> tuple[int, int, int, int]:
        return (
            int((box.x1 - cx1) * scale_x),
            int((box.y1 - cy1) * scale_y),
            int((box.x2 - cx1) * scale_x),
            int((box.y2 - cy1) * scale_y),
        )

    glow = Image.new("RGBA", left.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    tier_color = TIER_COLORS[claim.tier]
    for hit in claim.consensus.hits:
        glow_draw.rectangle(to_local(hit.box), outline=tier_color + (60,), width=3)
    left = Image.alpha_composite(left, glow)

    draw = ImageDraw.Draw(left)
    if claim.segmentation.segmented_box is not None:
        mask = claim.segmentation.mask
        seg_crop_box = claim.segmentation.crop_box
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [
                (
                    int((pt[0][0] + seg_crop_box.x1 - cx1) * scale_x),
                    int((pt[0][1] + seg_crop_box.y1 - cy1) * scale_y),
                )
                for pt in contour
            ]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill=SEGMENTATION_COLOR, width=2)

    draw.rectangle(to_local(claim.report_box), outline=tier_color, width=4)

    panel = Image.new("RGBA", (panel_width, left.height), PANEL_BG + (255,))
    pdraw = ImageDraw.Draw(panel)
    pad = 22
    pdraw.rectangle([0, 0, panel_width - 1, 46], fill=tier_color + (255,))
    pdraw.text((pad, 12), f"{claim.family.upper()} · {claim.tier.upper()}", font=font(18, bold=True), fill=(15, 15, 15))

    conformity = 1.0 - claim.nonconformity
    ring_cx, ring_cy, ring_r = panel_width - 66, 130, 44
    _draw_ring(pdraw, ring_cx, ring_cy, ring_r, conformity, tier_color)
    pdraw.text((ring_cx, ring_cy), f"{conformity*100:.0f}%", font=font(20, bold=True), fill=TEXT_PRIMARY, anchor="mm")
    pdraw.text((ring_cx, ring_cy + ring_r + 16), "trust score", font=font(12), fill=TEXT_SECONDARY, anchor="mm")

    rows = [
        ("Consensus agreement", f"{len({h.run_index for h in claim.consensus.hits})}/{claim.k_runs} runs"),
        ("Geometric tightness", f"{claim.signals.consensus_tightness:.2f} mean IoU"),
        ("Independent segmentation", f"{claim.signals.segmentation_agreement:.2f} IoU"),
        ("Colour/contrast plausibility", f"{claim.signals.radiometric_plausibility:.2f}"),
    ]
    y = 100
    label_x = pad
    for label, value in rows:
        pdraw.text((label_x, y), label, font=font(13), fill=TEXT_SECONDARY)
        pdraw.text((label_x, y + 18), value, font=font(16, bold=True), fill=TEXT_PRIMARY)
        y += 48

    composite = Image.new("RGBA", (left.width + panel.width, left.height), CARD_BG + (255,))
    composite.paste(left, (0, 0))
    composite.paste(panel, (left.width, 0))
    composite.convert("RGB").save(out_path)
