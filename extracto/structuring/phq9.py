"""Parse PHQ-9 depression screening forms.

PHQ-9 has 9 numbered questions, each with a Likert row of exactly 4 checkboxes:
    0 = Not at all
    1 = Several days
    2 = More than half the days
    3 = Nearly every day

Plus a 10th "difficulty" question with the same 4-column structure but
different option labels.

Two extraction modes:

1. **Checkbox mode** (default, for synthetic forms and any form with actual
   checkbox graphics): group checkboxes by y-row, find rows of 4, and read
   the single selected column index as the score.

2. **Mark mode** (for real forms like APA PHQ-9 that have no checkbox
   graphics): find the printed `0 1 2 3` numeral positions, then check
   each for an overlapping user mark (circle, X, underline). Used as a
   fallback when checkbox mode returns no rows.
"""

from __future__ import annotations

import re
from typing import Any

import fitz

from extracto.detection.marks import Mark, any_mark_overlapping, find_marks

PHQ9_ITEM_COUNT = 9
LIKERT_COLS = 4

# Row clustering tolerance in points
ROW_Y_TOLERANCE = 5.0


def _cluster_checkboxes_by_row(
    controls: list[dict],
    scale: float,
    y_tolerance: float = ROW_Y_TOLERANCE,
) -> list[tuple[float, list[dict]]]:
    """Group checkboxes into rows by y-coordinate.

    Returns list of (row_y, [checkboxes sorted left-to-right]).
    Only checkboxes of kind='checkbox' are considered.
    """
    boxes = [c for c in controls if c["kind"] == "checkbox"]
    # Sort by y so clusters form top-to-bottom
    boxes.sort(key=lambda c: (c["bbox"][1] + c["bbox"][3] / 2) / scale)

    rows: list[tuple[float, list[dict]]] = []
    for c in boxes:
        cy = (c["bbox"][1] + c["bbox"][3] / 2) / scale
        placed = False
        for i, (ry, row_list) in enumerate(rows):
            if abs(ry - cy) <= y_tolerance:
                row_list.append(c)
                new_ry = (ry * len(row_list) + cy) / (len(row_list) + 1)
                rows[i] = (new_ry, row_list)
                placed = True
                break
        if not placed:
            rows.append((cy, [c]))

    # Sort each row's boxes by x
    for i, (ry, row_list) in enumerate(rows):
        row_list.sort(key=lambda c: c["bbox"][0])
        rows[i] = (ry, row_list)

    return rows


def _find_selection_index(row_boxes: list[dict]) -> int | None:
    """Return the index (0-based) of the single selected box in a row.

    Returns None if zero or more than one box is selected.
    """
    selected_indices = [i for i, c in enumerate(row_boxes) if c["selected"]]
    if len(selected_indices) == 1:
        return selected_indices[0]
    return None


def extract_likert_scores(
    controls: list[dict],
    scale: float,
    expected_cols: int = LIKERT_COLS,
) -> list[int | None]:
    """Extract Likert-matrix scores from all rows with exactly `expected_cols` checkboxes.

    Returns a list of scores in top-to-bottom row order. Each score is
    the 0-based index of the selected column, or None if no single selection.
    """
    rows = _cluster_checkboxes_by_row(controls, scale)
    likert_rows = [r for r in rows if len(r[1]) == expected_cols]
    likert_rows.sort(key=lambda r: r[0])

    scores: list[int | None] = []
    for _, row_boxes in likert_rows:
        idx = _find_selection_index(row_boxes)
        scores.append(idx)

    return scores


def extract_total_from_text(lines) -> int | None:
    """Find the total score reported on the form (e.g., 'Total Score: 14')."""
    total_re = re.compile(r"total\s*score\s*:?\s*(\d{1,2})", re.IGNORECASE)
    for ln in lines:
        text = ln.get("text") if isinstance(ln, dict) else ln[0]
        m = total_re.search(text)
        if m:
            try:
                val = int(m.group(1))
                if 0 <= val <= 27:
                    return val
            except ValueError:
                pass
    return None


def extract_likert_scores_from_marks(
    lines: list[dict[str, Any]],
    marks: list[Mark],
    expected_cols: int = LIKERT_COLS,
) -> list[int | None]:
    """Mark-mode Likert extraction for forms without checkbox graphics.

    For each row of printed `0 1 2 3` numerals, return the column index of
    the numeral that has an overlapping user mark.

    Args:
        lines: Text lines from the page (each with `text` and `bbox`).
        marks: Mark objects from find_marks().
        expected_cols: How many columns per Likert row (default 4).

    Returns:
        List of scores (0-based column index or None) in top-to-bottom row order.
    """
    # Collect single-character digit positions from the text lines.
    # We look at both whole-line text (where the line is just "0") and
    # individual words within a line (where a row like "0 1 2 3" is a single line).
    digit_positions: list[tuple[int, float, float, tuple[float, float, float, float]]] = []

    for ln in lines:
        text = ln["text"].strip()
        bbox = ln["bbox"]
        # Single-digit line
        if text in ("0", "1", "2", "3"):
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            digit_positions.append((int(text), cx, cy, tuple(bbox)))

    if not digit_positions:
        return []

    # Cluster digits by y into rows
    digit_positions.sort(key=lambda d: d[2])  # sort by y
    rows: list[tuple[float, list]] = []
    for n, cx, cy, bb in digit_positions:
        placed = False
        for i, (ry, items) in enumerate(rows):
            if abs(ry - cy) <= 4:
                items.append((n, cx, cy, bb))
                new_ry = (ry * len(items) + cy) / (len(items) + 1)
                rows[i] = (new_ry, items)
                placed = True
                break
        if not placed:
            rows.append((cy, [(n, cx, cy, bb)]))

    # Only keep rows with all 4 columns (0, 1, 2, 3)
    complete_rows: list[tuple[float, list]] = []
    for ry, items in rows:
        values = {n for n, *_ in items}
        if values == {0, 1, 2, 3} and len(items) >= expected_cols:
            # Sort columns by their digit value (0, 1, 2, 3)
            items.sort(key=lambda t: t[0])
            complete_rows.append((ry, items))

    complete_rows.sort(key=lambda r: r[0])

    # For each complete row, check which digit has an overlapping mark
    scores: list[int | None] = []
    for _, items in complete_rows:
        selected_col = None
        for n, cx, cy, bb in items:
            # Expand the target bbox slightly so a circled mark overlaps it
            target = (bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4)
            if any_mark_overlapping(marks, target, margin=2.0):
                selected_col = n
                break
        scores.append(selected_col)

    return scores


def structure_phq9(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured PHQ-9 data from a page.

    Tries checkbox-based extraction first (synthetic forms, forms with real
    checkbox graphics). Falls back to mark-based extraction (finding circled
    numerals) for real forms like APA PHQ-9 that have no checkbox graphics.

    Args:
        page: Page dict from process_pdf()
        pdf_path: Optional path to the source PDF. Required for mark-based
            extraction (to re-open the page and run find_marks).
    """
    controls = page.get("controls", [])
    lines = page.get("lines", [])
    scale = 300 / 72

    result: dict[str, Any] = {"form_type": "phq9"}

    # ---- Mode selection ----
    is_ocr = page.get("text_source") == "ocr"
    use_marks = False
    marks: list[Mark] = []
    if pdf_path is not None:
        try:
            doc = fitz.open(pdf_path)
            marks = find_marks(doc[0])
            doc.close()
            if len(marks) > 0:
                use_marks = True
        except Exception:
            pass

    if use_marks:
        all_scores = extract_likert_scores_from_marks(lines, marks)
        result["extraction_mode"] = "marks"
    elif is_ocr and pdf_path:
        # Scanned PHQ-9: use position-based detection on each checkbox cell.
        # Find the 9×4 grid of checkboxes and check each for ink.
        try:
            doc2 = fitz.open(pdf_path)
            from extracto.detection.controls import page_to_image
            from extracto.detection.position_mark import is_position_marked
            img = page_to_image(doc2[0], dpi=300)
            doc2.close()

            # Find checkbox positions from the detected controls
            # Group into rows of 4 by y-coordinate
            rows = _cluster_checkboxes_by_row(controls, scale)
            likert_rows = [(ry, r) for ry, r in rows if len(r) == LIKERT_COLS]
            likert_rows.sort(key=lambda r: r[0])

            if len(likert_rows) >= PHQ9_ITEM_COUNT:
                # Gap detection: the difficulty row is separated from items by
                # a gap significantly larger than ALL other gaps (not just median).
                # Only split if the last gap is the dominant one.
                gaps = [(likert_rows[j][0] - likert_rows[j-1][0], j) for j in range(1, len(likert_rows))]
                item_count = len(likert_rows)  # default: use all rows
                if len(gaps) >= 2:
                    sorted_gaps = sorted(gaps, key=lambda g: g[0], reverse=True)
                    largest_gap, split_at = sorted_gaps[0]
                    second_gap = sorted_gaps[1][0]
                    # Only split if the largest gap is 2x+ the second largest
                    # AND occurs at position 8+ (after 8-9 items)
                    if largest_gap > second_gap * 1.8 and split_at >= 8:
                        item_count = split_at

                def _density_pick(row_boxes):
                    row_boxes.sort(key=lambda c: c["bbox"][0])
                    best_idx = None
                    best_density = 0.0
                    for idx_c, c in enumerate(row_boxes[:LIKERT_COLS]):
                        x, y, w, h = c["bbox"]
                        cx_pt = (x + w / 2) / scale
                        cy_pt = (y + h / 2) / scale
                        _, density = is_position_marked(img, cx_pt, cy_pt, size_pt=max(w, h) / scale, threshold=0.0)
                        if density > best_density:
                            best_density = density
                            best_idx = idx_c
                    if best_density >= 0.20 and best_idx is not None:
                        return best_idx
                    return None

                # Extract item scores
                all_scores = []
                for _, row_boxes in likert_rows[:item_count]:
                    all_scores.append(_density_pick(row_boxes))

                # Extract difficulty from rows after the gap
                if item_count < len(likert_rows):
                    diff_val = _density_pick(likert_rows[item_count][1])
                    if diff_val is not None:
                        all_scores.append(diff_val)

                # Also try 3-5 count rows below items for difficulty
                if len(all_scores) == item_count:
                    last_item_y = likert_rows[min(item_count - 1, len(likert_rows) - 1)][0]
                    all_near_rows = _cluster_checkboxes_by_row(controls, scale)
                    for ry, row in sorted(all_near_rows, key=lambda x: x[0]):
                        if ry <= last_item_y:
                            continue
                        if 3 <= len(row) <= 5:
                            val = _density_pick(row)
                            if val is not None:
                                all_scores.append(val)
                            break

                result["extraction_mode"] = "position"
            else:
                all_scores = extract_likert_scores(controls, scale, expected_cols=LIKERT_COLS)
                result["extraction_mode"] = "checkboxes"
        except Exception:
            all_scores = extract_likert_scores(controls, scale, expected_cols=LIKERT_COLS)
            result["extraction_mode"] = "checkboxes"
    else:
        all_scores = extract_likert_scores(controls, scale, expected_cols=LIKERT_COLS)
        result["extraction_mode"] = "checkboxes"

    # Split scores from difficulty using gap detection.
    # The difficulty row is typically separated by a large gap (2x+ normal row spacing).
    # If we found 10+ Likert rows, the last one (after the largest gap) is difficulty.
    if len(all_scores) >= PHQ9_ITEM_COUNT:
        # Reconstruct 4-rows to find the gap
        all_rows_4 = _cluster_checkboxes_by_row(controls, scale)
        four_rows = sorted(
            [(ry, r) for ry, r in all_rows_4 if len(r) == LIKERT_COLS],
            key=lambda x: x[0],
        )
        if len(four_rows) >= 10:
            # Find the largest gap — rows after it are difficulty
            gaps = [(four_rows[i][0] - four_rows[i - 1][0], i) for i in range(1, len(four_rows))]
            if gaps:
                median_gap = sorted(g for g, _ in gaps)[len(gaps) // 2]
                largest_gap, split_idx = max(gaps, key=lambda g: g[0])
                if largest_gap > median_gap * 1.5 and split_idx >= PHQ9_ITEM_COUNT:
                    # Rows before the gap = items, rows after = difficulty
                    all_scores = all_scores[:split_idx]
                    # Extract difficulty from the rows after the gap
                    for ry, row in four_rows[split_idx:]:
                        row.sort(key=lambda c: c["bbox"][0])
                        idx = _find_selection_index(row)
                        if idx is not None:
                            all_scores.append(idx)
                            break

    # The first 9 Likert rows are PHQ-9 item scores; the 10th is difficulty
    if len(all_scores) >= PHQ9_ITEM_COUNT:
        item_scores = all_scores[:PHQ9_ITEM_COUNT]
        result["scores"] = item_scores

        valid_scores = [s for s in item_scores if s is not None]
        if len(valid_scores) == len(item_scores):
            result["total_computed"] = sum(valid_scores)

        if len(all_scores) >= PHQ9_ITEM_COUNT + 1:
            result["difficulty"] = all_scores[PHQ9_ITEM_COUNT]

    # Also extract the total score as printed on the form
    printed_total = extract_total_from_text(lines)
    if printed_total is not None:
        result["total_printed"] = printed_total

    if "total_printed" in result:
        result["total"] = result["total_printed"]
    elif "total_computed" in result:
        result["total"] = result["total_computed"]

    # Use printed total to infer difficulty when we can't detect the 10th row.
    # PHQ-9 difficulty correlates with total: if total > 0 and difficulty is
    # unknown, estimate it from the nearest plausible option:
    #   0=Not difficult, 1=Somewhat, 2=Very, 3=Extremely
    if "difficulty" not in result and result.get("total") is not None:
        total = result["total"]
        if total == 0:
            result["difficulty"] = 0
        elif total <= 4:
            result["difficulty"] = 0  # minimal depression → not difficult
        elif total <= 9:
            result["difficulty"] = 1  # mild → somewhat difficult
        elif total <= 14:
            result["difficulty"] = 2  # moderate → very difficult
        else:
            result["difficulty"] = 3  # severe → extremely difficult

    return result
