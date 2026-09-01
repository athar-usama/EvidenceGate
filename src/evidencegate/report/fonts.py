"""Best-effort TrueType font loading with a safe fallback."""

from __future__ import annotations

from PIL import ImageFont

_CANDIDATES = [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
_BOLD_CANDIDATES = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _BOLD_CANDIDATES if bold else _CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
