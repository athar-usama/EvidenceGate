"""Renders a standalone SVG 'trust ring' for one claim, used instead of a bar/line chart."""

from __future__ import annotations

import math

TIER_COLORS = {
    "verified": "#1a9850",
    "flagged": "#d9a406",
    "abstained": "#c0392b",
}


def _arc_path(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> str:
    start_rad, end_rad = math.radians(start_deg), math.radians(end_deg)
    x1, y1 = cx + r * math.cos(start_rad), cy + r * math.sin(start_rad)
    x2, y2 = cx + r * math.cos(end_rad), cy + r * math.sin(end_rad)
    large_arc = 1 if (end_deg - start_deg) % 360 > 180 else 0
    return f"M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f}"


def trust_ring_svg(conformity: float, tier: str, size: int = 120, label: str | None = None) -> str:
    """conformity in [0, 1] (1 - nonconformity); the ring fills clockwise from the top."""
    conformity = max(0.0, min(1.0, conformity))
    cx, cy, r = size / 2, size / 2, size / 2 - 10
    color = TIER_COLORS.get(tier, "#888888")
    end_deg = -90 + 360 * conformity
    track = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e6e6e6" stroke-width="10"/>'
    if conformity >= 0.999:
        arc = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"/>'
    else:
        arc = (
            f'<path d="{_arc_path(cx, cy, r, -90, end_deg)}" fill="none" '
            f'stroke="{color}" stroke-width="10" stroke-linecap="round"/>'
        )
    text = (
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="{size*0.22:.0f}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{color}">{conformity*100:.0f}%</text>'
    )
    sublabel = (
        f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" font-size="{size*0.11:.0f}" '
        f'font-family="Segoe UI, Arial, sans-serif" fill="#555555">{label or tier.upper()}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f"{track}{arc}{text}{sublabel}</svg>"
    )


def write_trust_ring(path: str, conformity: float, tier: str, size: int = 120, label: str | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(trust_ring_svg(conformity, tier, size, label))
