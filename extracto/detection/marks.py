"""Detect user-added marks on real-world forms.

Many real forms (DWC-1, APA PHQ-9, clinic intakes) don't use checkbox graphics.
Users circle, X, or underline the option they want to select. This module
detects those marks and maps them to target positions.

Two detection modes:
1. **Color-based**: Marks with a distinctive stroke/fill color that differs
   from the black form template. Fast and precise for digitally-annotated PDFs.
2. **Template-diff** (future): Compare a filled page against a blank template
   and report new drawing elements. More robust but requires the blank form.

A "mark" is any drawing element that:
- Has a non-default color (not pure black) OR is flagged by template-diff
- Is small (< 40pt square) - marks are localized gestures, not form lines
- Has a bounding box - excludes color-only objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fitz


@dataclass
class Mark:
    """A detected user mark on a PDF page."""

    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    center: tuple[float, float]  # (cx, cy)
    color: tuple[float, float, float]  # RGB stroke/fill color
    kind: str  # 'stroke' or 'fill'
    size: float  # max(width, height) in points

    def overlaps(self, bbox: tuple[float, float, float, float], margin: float = 2.0) -> bool:
        """Check if this mark overlaps a target bounding box."""
        x0, y0, x1, y1 = bbox
        mx0, my0, mx1, my1 = self.bbox
        if mx1 < x0 - margin or mx0 > x1 + margin:
            return False
        if my1 < y0 - margin or my0 > y1 + margin:
            return False
        return True

    def contains_point(self, px: float, py: float, margin: float = 2.0) -> bool:
        x0, y0, x1, y1 = self.bbox
        return x0 - margin <= px <= x1 + margin and y0 - margin <= py <= y1 + margin

    def distance_to_point(self, px: float, py: float) -> float:
        cx, cy = self.center
        return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5


def _is_default_black(color: tuple | list | None) -> bool:
    """Is this a default black (form template) color?"""
    if not color:
        return True
    # Allow small floating-point noise around 0
    return all(c <= 0.05 for c in color[:3])


def find_marks(
    page: fitz.Page,
    *,
    max_size: float = 40.0,
    min_size: float = 2.0,
    exclude_colors: list[tuple] | None = None,
) -> list[Mark]:
    """Find user-added marks on a page.

    A drawing is considered a mark if:
    1. It has a non-black stroke or fill color
    2. Its bounding box fits within min_size / max_size
    3. It's not in the exclude list

    Args:
        page: PyMuPDF page
        max_size: Maximum size (in points) for a drawing to be considered a mark.
            Larger drawings are probably form lines or boxes.
        min_size: Minimum size (filters out text-rendering artifacts).
        exclude_colors: Additional colors to treat as default/template (e.g., for
            forms with grey grid lines).

    Returns:
        List of Mark objects sorted by (y, x).
    """
    exclude_set = set()
    if exclude_colors:
        for c in exclude_colors:
            exclude_set.add(tuple(round(x, 3) for x in c[:3]))

    drawings = page.get_drawings()
    marks: list[Mark] = []

    for d in drawings:
        stroke = d.get("color")
        fill = d.get("fill")
        rect = d.get("rect")
        if not rect:
            continue

        # Determine the effective color - prefer stroke, fall back to fill
        color: tuple | None = None
        kind = "stroke"
        if stroke and not _is_default_black(stroke):
            c3 = tuple(round(float(x), 3) for x in stroke[:3])
            if c3 not in exclude_set:
                color = c3
        if color is None and fill and not _is_default_black(fill):
            c3 = tuple(round(float(x), 3) for x in fill[:3])
            if c3 not in exclude_set:
                color = c3
                kind = "fill"
        if color is None:
            continue

        w = rect.x1 - rect.x0
        h = rect.y1 - rect.y0
        size = max(w, h)
        if size < min_size or size > max_size:
            continue

        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        marks.append(
            Mark(
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                center=(cx, cy),
                color=color,
                kind=kind,
                size=size,
            )
        )

    marks.sort(key=lambda m: (m.center[1], m.center[0]))
    return marks


def find_marks_near(marks: list[Mark], target_bbox: tuple[float, float, float, float], margin: float = 3.0) -> list[Mark]:
    """Return all marks that overlap a target bounding box."""
    return [m for m in marks if m.overlaps(target_bbox, margin=margin)]


def closest_mark(marks: list[Mark], target_cx: float, target_cy: float, max_distance: float = 12.0) -> Mark | None:
    """Return the closest mark to a target center point, or None if none within max_distance."""
    best = None
    best_dist = max_distance
    for m in marks:
        d = m.distance_to_point(target_cx, target_cy)
        if d < best_dist:
            best = m
            best_dist = d
    return best


def any_mark_overlapping(marks: list[Mark], target_bbox: tuple[float, float, float, float], margin: float = 3.0) -> bool:
    """Quick check: does any mark overlap the target bbox?"""
    for m in marks:
        if m.overlaps(target_bbox, margin=margin):
            return True
    return False


@dataclass
class OverlaidText:
    """A text span that was overlaid on top of a form template.

    Detected by having a non-black color, which distinguishes filled-in values
    from the printed form labels (which are always black in our fillers).
    """

    text: str
    bbox: tuple[float, float, float, float]
    color: tuple[int, int, int]  # 8-bit RGB

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)


def find_overlaid_text(page: fitz.Page) -> list[OverlaidText]:
    """Find text spans with a non-default (non-black) color on a page.

    Used to distinguish user-added form values from printed form template text.
    Assumes the form template is rendered in black and filled values are
    rendered in a distinctive color (e.g., dark blue).
    """
    text_dict = page.get_text("dict")
    out: list[OverlaidText] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                color = span.get("color", 0)
                if color == 0:
                    continue
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                text = span.get("text", "").strip()
                if not text:
                    continue
                out.append(OverlaidText(
                    text=text,
                    bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                    color=(r, g, b),
                ))
    return out


def nearest_overlaid_text(
    overlays: list[OverlaidText],
    anchor_bbox: tuple[float, float, float, float],
    *,
    direction: str = "below",
    max_dy: float = 25,
    max_dx: float = 250,
) -> OverlaidText | None:
    """Find the overlaid text span nearest to an anchor, in a given direction.

    Args:
        overlays: List from find_overlaid_text()
        anchor_bbox: The label's bbox
        direction: 'below' (looks directly below the anchor) or 'right' (same row)
        max_dy: Max vertical distance
        max_dx: Max horizontal distance
    """
    ax0, ay0, ax1, ay1 = anchor_bbox
    anchor_cx = (ax0 + ax1) / 2
    anchor_cy = (ay0 + ay1) / 2

    best = None
    best_score = 1e9

    for ot in overlays:
        ox0, oy0, ox1, oy1 = ot.bbox
        ocx = (ox0 + ox1) / 2
        ocy = (oy0 + oy1) / 2

        if direction == "below":
            dy = ocy - ay1
            if dy < -2 or dy > max_dy:
                continue
            # Horizontal alignment: the text should start near the anchor's x range
            if ox0 < ax0 - 20 or ox0 > ax1 + max_dx:
                continue
            score = dy + abs(ox0 - ax0) * 0.1
        elif direction == "right":
            # Allow the overlay to start slightly left of the anchor's right edge
            # (some fillers render text with baseline anchoring that pushes the
            # bbox left of the expected position).
            dx = ox0 - ax1
            if dx < -30 or dx > max_dx:
                continue
            if abs(ocy - anchor_cy) > max_dy:
                continue
            # Prefer positive dx (actually to the right) but allow small negatives
            score = abs(dx) + abs(ocy - anchor_cy) * 0.5
        else:
            continue

        if score < best_score:
            best = ot
            best_score = score

    return best
