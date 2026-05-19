"""Position-based mark detection — for scanned forms where we know where checkboxes
should be (relative to OCR'd anchor text) but contour-based detection fails.

Workflow:
1. The structuring layer knows "the Yes checkbox should be just left of the YES text"
2. It calls is_position_marked(image, x, y, size) to ask: "is there a mark here?"
3. We crop the image at that position and measure ink density

This bypasses general checkbox detection entirely. We don't try to find every
checkbox in the image — we only check the specific positions where structured
forms tell us checkboxes should be.

Used as a fallback when conventional CV detection produces too few controls.
"""

from __future__ import annotations

import cv2
import numpy as np


def is_position_marked(
    img: np.ndarray,
    x_pt: float,
    y_pt: float,
    size_pt: float = 12.0,
    dpi: int = 300,
    threshold: float = 0.15,
) -> tuple[bool, float]:
    """Check if there's a mark (filled checkbox / X / check) at the given position.

    Args:
        img: Page raster (BGR, full page at `dpi` resolution)
        x_pt, y_pt: Center position in PDF points (top-left origin)
        size_pt: Expected checkbox size in points (default 12pt)
        dpi: Image DPI
        threshold: Ink density threshold (0-1) above which the position is "marked"

    Returns:
        (is_marked, confidence) where confidence is the measured ink density.
    """
    scale = dpi / 72.0
    cx = int(x_pt * scale)
    cy = int(y_pt * scale)
    half_size = int(size_pt * scale / 2)

    h, w = img.shape[:2]
    x0 = max(0, cx - half_size)
    y0 = max(0, cy - half_size)
    x1 = min(w, cx + half_size)
    y1 = min(h, cy + half_size)

    if x1 - x0 < 4 or y1 - y0 < 4:
        return False, 0.0

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Threshold to binary (dark pixels = ink)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # Compute ink density in the INTERIOR (exclude border, which is the
    # checkbox outline itself — we want to know if the box is FILLED, not
    # whether there's a box there).
    h_crop, w_crop = binary.shape
    margin = max(2, int(min(h_crop, w_crop) * 0.20))
    interior = binary[margin : h_crop - margin, margin : w_crop - margin]
    if interior.size == 0:
        return False, 0.0

    density = float(np.count_nonzero(interior)) / interior.size
    return (density >= threshold, density)


def find_marks_at_positions(
    img: np.ndarray,
    positions: list[tuple[float, float, str]],
    size_pt: float = 12.0,
    dpi: int = 300,
    threshold: float = 0.15,
) -> dict[str, tuple[bool, float]]:
    """Check multiple positions at once. Returns {label: (is_marked, density)}."""
    results = {}
    for x_pt, y_pt, label in positions:
        results[label] = is_position_marked(img, x_pt, y_pt, size_pt, dpi, threshold)
    return results
