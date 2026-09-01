"""Box representation and geometric utilities shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in absolute pixel coordinates, (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0.0, min(self.x1, width - 1)),
            y1=max(0.0, min(self.y1, height - 1)),
            x2=max(0.0, min(self.x2, width - 1)),
            y2=max(0.0, min(self.y2, height - 1)),
        )

    def expand(self, margin_frac: float, width: int, height: int) -> "Box":
        """Pad a box by a fraction of its own size, clipped to image bounds."""
        w, h = self.x2 - self.x1, self.y2 - self.y1
        dx, dy = w * margin_frac, h * margin_frac
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clip(width, height)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance_norm(a: Box, b: Box) -> float:
    """Center distance normalized by the mean box diagonal — used when boxes barely overlap."""
    ax, ay = a.center()
    bx, by = b.center()
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    diag_a = ((a.x2 - a.x1) ** 2 + (a.y2 - a.y1) ** 2) ** 0.5
    diag_b = ((b.x2 - b.x1) ** 2 + (b.y2 - b.y1) ** 2) ** 0.5
    mean_diag = max(1e-6, (diag_a + diag_b) / 2.0)
    return dist / mean_diag


def union_box(boxes: list[Box]) -> Box:
    return Box(
        x1=min(b.x1 for b in boxes),
        y1=min(b.y1 for b in boxes),
        x2=max(b.x2 for b in boxes),
        y2=max(b.y2 for b in boxes),
    )


def mean_box(boxes: list[Box]) -> Box:
    n = len(boxes)
    return Box(
        x1=sum(b.x1 for b in boxes) / n,
        y1=sum(b.y1 for b in boxes) / n,
        x2=sum(b.x2 for b in boxes) / n,
        y2=sum(b.y2 for b in boxes) / n,
    )
