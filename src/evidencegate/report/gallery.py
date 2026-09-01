"""Full-image annotation and grid/side-by-side composition for the README galleries."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evidencegate.pipeline import Claim, ImageEvidence, TIER_ABSTAINED, TIER_FLAGGED, TIER_VERIFIED
from evidencegate.report.evidence_card import TIER_COLORS
from evidencegate.report.fonts import font

_LINE_STYLE = {TIER_VERIFIED: 5, TIER_FLAGGED: 3, TIER_ABSTAINED: 1}


def annotate_image(
    bgr: np.ndarray, claims: list[Claim], tiers_to_show: tuple[str, ...] = (TIER_VERIFIED, TIER_FLAGGED)
) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for claim in claims:
        if claim.tier not in tiers_to_show:
            continue
        color = TIER_COLORS[claim.tier]
        width = _LINE_STYLE[claim.tier]
        box = claim.report_box
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=width)
    return image


def _fit_text(text: str, max_width: int, base_size: int, bold: bool = True, min_size: int = 11) -> tuple[str, "ImageFont.FreeTypeFont"]:
    """Shrinks font size, then truncates with an ellipsis, until `text` fits `max_width`."""
    for size in range(base_size, min_size - 1, -1):
        f = font(size, bold=bold)
        if f.getlength(text) <= max_width:
            return text, f
    f = font(min_size, bold=bold)
    truncated = text
    while truncated and f.getlength(truncated + "...") > max_width:
        truncated = truncated[:-1]
    return (truncated + "..." if truncated != text else text), f


def add_caption(image: Image.Image, text: str, bar_height: int = 30, bg=(18, 20, 24), fg=(235, 236, 240)) -> Image.Image:
    captioned = Image.new("RGB", (image.width, image.height + bar_height), bg)
    captioned.paste(image, (0, 0))
    draw = ImageDraw.Draw(captioned)
    fitted_text, fitted_font = _fit_text(text, image.width - 16, base_size=15)
    draw.text((8, image.height + bar_height // 2), fitted_text, font=fitted_font, fill=fg, anchor="lm")
    return captioned


def _resize_to_fit(image: Image.Image, size: int) -> Image.Image:
    """Scales (up or down) so the longer side equals `size`, preserving aspect ratio."""
    scale = size / max(image.width, image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.LANCZOS if scale < 1 else Image.NEAREST
    return image.resize(new_size, resample)


def make_grid(images: list[Image.Image], cols: int, cell_size: int = 260, pad: int = 8, bg=(10, 11, 13)) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_size + (cols + 1) * pad, rows * cell_size + (rows + 1) * pad), bg)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        thumb = _resize_to_fit(img, cell_size)
        x = pad + c * (cell_size + pad) + (cell_size - thumb.width) // 2
        y = pad + r * (cell_size + pad) + (cell_size - thumb.height) // 2
        grid.paste(thumb, (x, y))
    return grid


def side_by_side(left: Image.Image, right: Image.Image, left_label: str, right_label: str, gap: int = 12, panel_size: int = 260) -> Image.Image:
    left_sq = add_caption(_resize_to_fit(left, panel_size), left_label)
    right_sq = add_caption(_resize_to_fit(right, panel_size), right_label)
    canvas = Image.new("RGB", (left_sq.width + right_sq.width + gap, max(left_sq.height, right_sq.height)), (10, 11, 13))
    canvas.paste(left_sq, (0, 0))
    canvas.paste(right_sq, (left_sq.width + gap, 0))
    return canvas
