"""PDF-native control detection using vector drawing information.

For digitally-generated PDFs, the checkbox and radio button rectangles exist
as vector drawings we can extract directly. This is far more accurate than
rasterizing the page and running computer vision.

Drawings classification:
- type='s' with items containing ONLY 'c' (curves) → circle (radio outline)
- type='s' with items containing 're' OR ('l' + 'c' mix) → rectangle (checkbox outline,
  possibly with rounded corners)
- type='f' drawings → selection markers (filled interior rectangle or circle)
- Individual 'l' (line) drawings inside a checkbox → check/X mark

Used as the primary detection path when the PDF has vector drawings.
Falls back to CV detection for scanned/raster-only PDFs.
"""

from __future__ import annotations

from typing import Any

import fitz


CHECKBOX_MIN_SIZE = 5.0  # points - CMS-1500 uses small boxes for SSN/EIN
CHECKBOX_MAX_SIZE = 22.0
CHECKBOX_MAX_ASPECT_DIFF = 3.0


def _is_square(rect: fitz.Rect) -> bool:
    w = rect.x1 - rect.x0
    h = rect.y1 - rect.y0
    if w < CHECKBOX_MIN_SIZE or w > CHECKBOX_MAX_SIZE:
        return False
    if h < CHECKBOX_MIN_SIZE or h > CHECKBOX_MAX_SIZE:
        return False
    if abs(w - h) > CHECKBOX_MAX_ASPECT_DIFF:
        return False
    return True


def _rect_contains_point(outer: fitz.Rect, px: float, py: float, margin: float = 1.0) -> bool:
    return (
        outer.x0 - margin <= px <= outer.x1 + margin
        and outer.y0 - margin <= py <= outer.y1 + margin
    )


def _rect_contains(outer: fitz.Rect, inner: fitz.Rect, margin: float = 1.0) -> bool:
    return (
        inner.x0 >= outer.x0 - margin
        and inner.x1 <= outer.x1 + margin
        and inner.y0 >= outer.y0 - margin
        and inner.y1 <= outer.y1 + margin
    )


def _classify_drawing(d: dict) -> str | None:
    """Return 'circle', 'rect', or None based on the drawing's item pattern."""
    items = d.get("items", [])
    if not items:
        return None

    ops = [it[0] for it in items]

    # Pure 're' rectangle
    if "re" in ops:
        return "rect"

    # Only curves (circles drawn as 4 bezier curves)
    has_curves = "c" in ops
    has_lines = "l" in ops

    if has_curves and not has_lines:
        return "circle"

    # Mix of lines and curves suggests rounded rectangle
    if has_curves and has_lines:
        # A rounded rect has 4 straight edges and 4 corner curves
        line_count = sum(1 for o in ops if o == "l")
        curve_count = sum(1 for o in ops if o == "c")
        if line_count >= 3 and curve_count >= 3:
            return "rect"
        return None

    # Pure lines might be a polygon/rect
    if has_lines and not has_curves:
        line_count = sum(1 for o in ops if o == "l")
        if line_count >= 3:
            return "rect"

    return None


def extract_controls_from_drawings(
    page: fitz.Page,
    scale: float = 300 / 72,
) -> list[dict[str, Any]]:
    """Extract controls by parsing PDF vector drawings.

    Returns a list of control dicts compatible with the CV pipeline output:
        {"kind": "checkbox"|"radio", "bbox": (x, y, w, h) in pixels,
         "selected": bool, "label": "", "conf": float}

    Returns empty list if the PDF has no useful drawings.
    """
    drawings = page.get_drawings()
    if not drawings:
        return []

    # Phase 1: classify each drawing
    checkbox_outlines: list[fitz.Rect] = []
    radio_outlines: list[fitz.Rect] = []
    filled_rects: list[fitz.Rect] = []
    filled_circles: list[fitz.Rect] = []  # rects of filled circle drawings
    # Standalone line segments (candidates for checkmarks inside a checkbox)
    standalone_lines: list[tuple[tuple, tuple]] = []

    for d in drawings:
        rect = d.get("rect")
        d_type = d.get("type")
        items = d.get("items", [])

        if not rect or not isinstance(rect, fitz.Rect):
            continue

        shape = _classify_drawing(d)

        if d_type == "s" and _is_square(rect):
            if shape == "rect":
                checkbox_outlines.append(rect)
            elif shape == "circle":
                radio_outlines.append(rect)
        elif d_type == "f":
            if _is_square(rect):
                if shape == "circle":
                    filled_circles.append(rect)
                elif shape == "rect":
                    filled_rects.append(rect)
            else:
                # Small filled shapes can be selection marks (inner rect/dot)
                w = rect.x1 - rect.x0
                h = rect.y1 - rect.y0
                if 1 <= w <= 20 and 1 <= h <= 20:
                    if shape == "circle":
                        filled_circles.append(rect)
                    else:
                        filled_rects.append(rect)

        # Track standalone lines (for checkmark detection)
        if d_type == "s" and len(items) <= 2:
            for item in items:
                if item[0] == "l":
                    standalone_lines.append((item[1], item[2]))

    controls: list[dict[str, Any]] = []

    # Phase 2: build checkbox controls
    seen_checkbox_keys: set[tuple[int, int]] = set()
    for box in checkbox_outlines:
        key = (int(box.x0 * 10), int(box.y0 * 10))
        if key in seen_checkbox_keys:
            continue
        seen_checkbox_keys.add(key)

        selected = False

        # Check for filled interior rectangle
        for f in filled_rects:
            if _rect_contains(box, f, margin=1.0):
                selected = True
                break

        # Check for check/X mark lines inside the box
        if not selected:
            lines_inside = 0
            for p1, p2 in standalone_lines:
                # Both endpoints must be inside the box
                if (
                    _rect_contains_point(box, p1[0], p1[1], margin=1.0)
                    and _rect_contains_point(box, p2[0], p2[1], margin=1.0)
                ):
                    lines_inside += 1
            if lines_inside >= 2:
                # Two line segments = check mark or X
                selected = True
            elif lines_inside == 1:
                # A single diagonal line might still be a check mark component,
                # but be conservative and only count if no filled test matched
                pass

        x = int(box.x0 * scale)
        y = int(box.y0 * scale)
        w = int((box.x1 - box.x0) * scale)
        h = int((box.y1 - box.y0) * scale)
        controls.append({
            "kind": "checkbox",
            "bbox": (x, y, w, h),
            "selected": selected,
            "label": "",
            "conf": 0.98 if selected else 0.95,
        })

    # Phase 3: build radio controls
    seen_radio_keys: set[tuple[int, int]] = set()
    for rbox in radio_outlines:
        key = (int(rbox.x0 * 10), int(rbox.y0 * 10))
        if key in seen_radio_keys:
            continue
        seen_radio_keys.add(key)

        selected = False

        # A selected radio has a filled circle inside
        center_x = (rbox.x0 + rbox.x1) / 2
        center_y = (rbox.y0 + rbox.y1) / 2
        radius = (rbox.x1 - rbox.x0) / 2

        for fc in filled_circles:
            fc_cx = (fc.x0 + fc.x1) / 2
            fc_cy = (fc.y0 + fc.y1) / 2
            dist = ((fc_cx - center_x) ** 2 + (fc_cy - center_y) ** 2) ** 0.5
            fc_radius = (fc.x1 - fc.x0) / 2
            if dist <= radius * 0.5 and fc_radius < radius * 0.9:
                selected = True
                break

        # Fallback: check for filled rect inside (some filled renders are rects)
        if not selected:
            for f in filled_rects:
                if _rect_contains(rbox, f, margin=1.0):
                    selected = True
                    break

        x = int(rbox.x0 * scale)
        y = int(rbox.y0 * scale)
        w = int((rbox.x1 - rbox.x0) * scale)
        h = int((rbox.y1 - rbox.y0) * scale)
        controls.append({
            "kind": "radio",
            "bbox": (x, y, w, h),
            "selected": selected,
            "label": "",
            "conf": 0.98 if selected else 0.95,
        })

    return controls


def has_useful_drawings(page: fitz.Page) -> bool:
    """Check if the page has vector drawings suitable for native extraction."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return False

    checkbox_like_count = 0
    for d in drawings:
        rect = d.get("rect")
        if rect and isinstance(rect, fitz.Rect) and _is_square(rect):
            checkbox_like_count += 1
            if checkbox_like_count >= 3:
                return True

    return False
