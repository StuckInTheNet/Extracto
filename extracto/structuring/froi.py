"""Parse FROI (First Report of Injury) workers compensation forms.

Distinctive features:
- Large body parts grid with **left/right laterality** (Shoulder-L, Knee-R, etc.)
- Multiple multi-select sections (causes, treatments)
- Single-select nature of injury
- Text fields for employee/employer info

Two extraction modes:
1. **Checkbox mode** for synthetic forms with rectangle graphics
2. **Mark mode** for real forms (like Florida DWC-1) that print letters like
   'M F' or 'YES NO' and expect the user to circle/X their choice. Falls back
   to mark-mode when no checkbox graphics are found.
"""

from __future__ import annotations

import re
from typing import Any

import fitz

from extracto.detection.marks import (
    Mark,
    OverlaidText,
    any_mark_overlapping,
    find_marks,
    find_overlaid_text,
    nearest_overlaid_text,
)

# Match the generator's BODY_PARTS list
BODY_PARTS: list[tuple[str, bool]] = [
    ("Head", False),
    ("Eye", True),
    ("Ear", True),
    ("Face", False),
    ("Neck", False),
    ("Shoulder", True),
    ("Upper Arm", True),
    ("Elbow", True),
    ("Forearm", True),
    ("Wrist", True),
    ("Hand", True),
    ("Fingers", True),
    ("Chest", False),
    ("Back-Upper", False),
    ("Back-Lower", False),
    ("Abdomen", False),
    ("Hip", True),
    ("Thigh", True),
    ("Knee", True),
    ("Lower Leg", True),
    ("Ankle", True),
    ("Foot", True),
    ("Toes", True),
    ("Internal", False),
    ("Multiple", False),
]

NATURE_OF_INJURY = [
    "Strain/Sprain",
    "Fracture",
    "Laceration",
    "Contusion",
    "Burn",
    "Dislocation",
    "Puncture",
    "Concussion",
    "Other",
]

CAUSE_OF_INJURY = [
    "Fall - Same Level",
    "Fall - From Height",
    "Struck By Object",
    "Struck Against Object",
    "Caught In/Between",
    "Repetitive Motion",
    "Lifting",
    "Motor Vehicle",
    "Chemical Exposure",
    "Other",
]

TREATMENT_OPTIONS = [
    "First Aid Only",
    "Clinic/Doctor Visit",
    "Emergency Room",
    "Hospitalized",
    "None",
]

DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")


# --- Helpers ---

def _find_line_contains(lines, needle: str):
    """Multi-tier match: exact → whitespace-tolerant → fuzzy (for OCR'd text)."""
    from extracto.structuring.fuzzy import fuzzy_contains, normalize
    needle_norm = normalize(needle)
    needle_nospace = needle_norm.replace(" ", "")
    for ln in lines:
        text = normalize(ln["text"])
        if needle_norm in text or needle_nospace in text.replace(" ", ""):
            return ln
    for ln in lines:
        if fuzzy_contains(ln["text"], needle, threshold=0.78):
            return ln
    return None


def _nearest_text_right(lines, label_text: str, max_dx: float = 300, max_dy: float = 4) -> str | None:
    anchor = _find_line_contains(lines, label_text)
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    band_cy = (ay0 + ay1) / 2

    best = None
    best_dx = 1e9
    for ln in lines:
        if ln is anchor:
            continue
        bbox = ln["bbox"]
        lcy = (bbox[1] + bbox[3]) / 2
        if abs(lcy - band_cy) > max_dy:
            continue
        dx = bbox[0] - ax1
        if 0 <= dx <= max_dx and dx < best_dx:
            # Skip other label fields
            if ":" in ln["text"] and ":" not in label_text:
                continue
            best = ln["text"].strip()
            best_dx = dx
    return best


def _find_checkbox_for_label(
    controls,
    lines,
    label_text: str,
    scale: float,
    section_y_bounds: tuple[float, float] | None = None,
    max_dx: float = 20,
) -> dict | None:
    """Find the checkbox immediately to the left of a label within an optional y bounds."""
    label_line = None
    for ln in lines:
        if label_text.lower() == ln["text"].strip().lower() or f" {label_text.lower()}" in f" {ln['text'].lower()} ":
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            if section_y_bounds and not (section_y_bounds[0] <= cy <= section_y_bounds[1]):
                continue
            label_line = ln
            break
    if not label_line:
        return None

    lx0, ly0, lx1, ly1 = label_line["bbox"]
    lcy = (ly0 + ly1) / 2
    best = None
    best_dx = 1e9
    for c in controls:
        if c["kind"] != "checkbox":
            continue
        x, y, w, h = c["bbox"]
        cy = (y + h / 2) / scale
        cx_right = (x + w) / scale
        if abs(cy - lcy) > 7:
            continue
        dx = lx0 - cx_right
        if 0 <= dx <= max_dx and dx < best_dx:
            best = c
            best_dx = dx
    return best


def _section_y_bounds(lines, section_label: str, next_section_labels: list[str]) -> tuple[float, float] | None:
    """Return (y_start, y_end) for a labeled section."""
    start = _find_line_contains(lines, section_label)
    if not start:
        return None
    y_start = start["bbox"][3]

    y_end = 99999.0
    for next_label in next_section_labels:
        nl = _find_line_contains(lines, next_label)
        if nl:
            ny = nl["bbox"][1]
            if y_start < ny < y_end:
                y_end = ny
    return y_start, y_end


# --- Extractors ---

def extract_employee_name(lines) -> str | None:
    val = _nearest_text_right(lines, "employee name", max_dx=300, max_dy=5)
    if val:
        return val.rstrip(":")
    return None


def extract_employer_name(lines) -> str | None:
    val = _nearest_text_right(lines, "employer name", max_dx=300, max_dy=5)
    if val:
        return val.rstrip(":")
    return None


def extract_injury_date(lines) -> str | None:
    """Find the injury date - the first date following 'Date of Injury' label."""
    anchor = _find_line_contains(lines, "date of injury")
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    band_cy = (ay0 + ay1) / 2

    for ln in lines:
        bbox = ln["bbox"]
        lcy = (bbox[1] + bbox[3]) / 2
        if abs(lcy - band_cy) > 4:
            continue
        if bbox[0] < ax1:
            continue
        m = DATE_RE.search(ln["text"])
        if m:
            return m.group(1)
    return None


def extract_on_premises(lines, controls, scale) -> bool | None:
    """Find the Yes/No pair for 'On employer premises?'."""
    anchor = _find_line_contains(lines, "on employer premises")
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    band_cy = (ay0 + ay1) / 2

    # Find Yes/No checkboxes to the right of the anchor
    candidates = []
    for c in controls:
        if c["kind"] != "checkbox":
            continue
        x, y, w, h = c["bbox"]
        cy = (y + h / 2) / scale
        cx = (x + w / 2) / scale
        if abs(cy - band_cy) > 7:
            continue
        if ax1 - 5 <= cx <= ax1 + 200:
            candidates.append(c)

    # Use labeled Yes/No first
    yes_labeled = [c for c in candidates if (c.get("label") or "").strip().lower().startswith("yes")]
    no_labeled = [c for c in candidates if (c.get("label") or "").strip().lower().startswith("no")]
    yes_sel = any(c["selected"] for c in yes_labeled)
    no_sel = any(c["selected"] for c in no_labeled)

    if not yes_labeled or not no_labeled:
        if len(candidates) >= 2:
            candidates.sort(key=lambda c: c["bbox"][0])
            yes_sel = candidates[0]["selected"]
            no_sel = candidates[1]["selected"]

    if yes_sel and not no_sel:
        return True
    if no_sel and not yes_sel:
        return False
    return None


def extract_injured_body_parts(lines, controls, scale) -> list[str]:
    """Extract the set of injured body parts with laterality markers.

    Layout: each body part occupies a row with:
    - Name (e.g. "Shoulder") on the left
    - 1 checkbox (no laterality) or 2 checkboxes labeled L and R (with laterality)

    We find each body part's text line, then look for checkboxes on the same row
    to its right.
    """
    # Get the section bounds
    bounds = _section_y_bounds(
        lines,
        "body parts injured",
        ["nature of injury", "cause of injury", "treatment received"],
    )
    if not bounds:
        return []
    y_start, y_end = bounds

    injured: list[str] = []

    for part, has_lat in BODY_PARTS:
        # Find the text line for this body part within the section
        part_line = None
        for ln in lines:
            text = ln["text"].strip()
            if text == part:
                cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
                if y_start <= cy <= y_end:
                    part_line = ln
                    break
        if not part_line:
            continue

        lx0, ly0, lx1, ly1 = part_line["bbox"]
        lcy = (ly0 + ly1) / 2

        # Find checkboxes on the same row, within a horizontal window right of the name
        row_checkboxes = []
        for c in controls:
            if c["kind"] != "checkbox":
                continue
            x, y, w, h = c["bbox"]
            cy = (y + h / 2) / scale
            cx = (x + w / 2) / scale
            if abs(cy - lcy) > 6:
                continue
            if lx1 <= cx <= lx1 + 120:
                row_checkboxes.append((cx, c))

        row_checkboxes.sort(key=lambda t: t[0])

        if has_lat:
            # Expected: 2 checkboxes (L then R)
            if len(row_checkboxes) >= 2:
                l_box = row_checkboxes[0][1]
                r_box = row_checkboxes[1][1]
                if l_box["selected"]:
                    injured.append(f"{part}-L")
                if r_box["selected"]:
                    injured.append(f"{part}-R")
        else:
            # Expected: 1 checkbox
            if row_checkboxes and row_checkboxes[0][1]["selected"]:
                injured.append(part)

    return sorted(injured)


def extract_nature_of_injury(lines, controls, scale) -> str | None:
    """Single-select: find the checked option in the NATURE OF INJURY section."""
    bounds = _section_y_bounds(
        lines,
        "nature of injury",
        ["cause of injury", "treatment received"],
    )
    if not bounds:
        return None

    for nature in NATURE_OF_INJURY:
        box = _find_checkbox_for_label(
            controls, lines, nature, scale, section_y_bounds=bounds
        )
        if box and box["selected"]:
            return nature
    return None


def extract_causes(lines, controls, scale) -> list[str]:
    """Multi-select: find all checked options in CAUSE OF INJURY section."""
    bounds = _section_y_bounds(
        lines,
        "cause of injury",
        ["treatment received"],
    )
    if not bounds:
        return []

    selected: list[str] = []
    for cause in CAUSE_OF_INJURY:
        box = _find_checkbox_for_label(
            controls, lines, cause, scale, section_y_bounds=bounds
        )
        if box and box["selected"]:
            selected.append(cause)
    return sorted(selected)


def extract_treatments(lines, controls, scale) -> list[str]:
    """Multi-select: find all checked options in TREATMENT RECEIVED section."""
    bounds = _section_y_bounds(
        lines,
        "treatment received",
        [],  # last section
    )
    if not bounds:
        return []
    # Extend to the bottom of the page since there's no next section
    bounds = (bounds[0], bounds[0] + 100)

    selected: list[str] = []
    for t in TREATMENT_OPTIONS:
        box = _find_checkbox_for_label(
            controls, lines, t, scale, section_y_bounds=bounds
        )
        if box and box["selected"]:
            selected.append(t)
    return sorted(selected)


def extract_marked_letter(
    lines,
    marks: list[Mark],
    anchor_text: str,
    candidates: list[str],
    max_dy: float = 25,
    max_dx: float = 150,
) -> str | None:
    """Mark-mode extraction: find which of N candidate letters/words has a user mark.

    Real forms like DWC-1 print answer options as raw letters (e.g., 'M F',
    'YES NO', 'AM PM') that the user is supposed to circle. This helper:

    1. Finds the anchor text position
    2. For each candidate letter/word, finds its position in a window right
       of the anchor
    3. Checks each candidate for an overlapping mark
    4. Returns the first marked candidate (or None)
    """
    anchor = _find_line_contains(lines, anchor_text)
    if not anchor:
        return None
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    acy = (ay0 + ay1) / 2

    for candidate in candidates:
        # Find the candidate token in proximity to the anchor
        for ln in lines:
            text = ln["text"].strip()
            if text != candidate:
                continue
            bbox = ln["bbox"]
            lcx = (bbox[0] + bbox[2]) / 2
            lcy = (bbox[1] + bbox[3]) / 2
            # Accept if within vertical tolerance AND right of / near the anchor
            if abs(lcy - acy) > max_dy:
                continue
            if lcx < ax0 - 20 or lcx > ax1 + max_dx:
                continue
            # Check for an overlapping mark on this candidate
            target = (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4)
            if any_mark_overlapping(marks, target, margin=2.0):
                return candidate
            break  # found the candidate position but no mark - try next candidate
    return None


def structure_froi_real(page: dict[str, Any], pdf_path: str) -> dict[str, Any]:
    """Extract fields from a REAL-WORLD FROI / DWC-1 form using mark detection.

    Real DWC-1 forms have no checkbox graphics. Users circle printed letters
    (M/F, YES/NO) or write in blank underlines. This extractor:
    - Uses mark detection to read circled-letter fields
    - Uses colored-text detection to read filled text overlays (distinguishing
      user-entered values from printed form labels by color)
    """
    lines = page.get("lines", [])

    result: dict[str, Any] = {"form_type": "froi", "extraction_mode": "marks"}

    # Open the PDF to get drawings + overlaid text + tight anchor bboxes
    fitz_page = None
    try:
        doc = fitz.open(pdf_path)
        fitz_page = doc[0]
        marks = find_marks(fitz_page)
        overlays = find_overlaid_text(fitz_page)
    except Exception:
        marks = []
        overlays = []

    result["marks_detected"] = len(marks)
    result["overlays_detected"] = len(overlays)

    # --- Overlaid text fields (color-distinguished from form template) ---
    def extract_overlay_field(anchor_text: str, direction: str = "below", max_dy: float = 25, max_dx: float = 250) -> str | None:
        """Use page.search_for() to get a tight bbox for the anchor phrase only
        (not the full grouped text line, which can include trailing underscores).
        """
        if fitz_page is None:
            return None
        rects = fitz_page.search_for(anchor_text)
        if not rects:
            return None
        # Use the first match
        anchor_bbox = tuple(rects[0])
        ot = nearest_overlaid_text(
            overlays, anchor_bbox, direction=direction, max_dy=max_dy, max_dx=max_dx
        )
        return ot.text if ot else None

    name_val = extract_overlay_field("name (first", direction="below", max_dy=20)
    if name_val:
        result["employee_name"] = name_val

    # Accident date - overlaid below the "Date of Accident" anchor
    date_val = extract_overlay_field("date of accident", direction="below", max_dy=20)
    if date_val and re.match(r"\d{2}/\d{2}/\d{4}", date_val):
        result["accident_date"] = date_val

    # DOB - overlaid below the "DATE OF BIRTH" anchor
    dob_val = extract_overlay_field("date of birth", direction="below", max_dy=20)
    if dob_val and re.match(r"\d{2}/\d{2}/\d{4}", dob_val):
        result["employee_dob"] = dob_val

    occ_val = extract_overlay_field("occupation", direction="below", max_dy=20)
    if occ_val:
        result["occupation"] = occ_val

    # Company name - overlaid to the RIGHT of "COMPANY NAME:"
    company_val = extract_overlay_field("company name", direction="right", max_dy=8, max_dx=300)
    if company_val:
        result["company_name"] = company_val

    body_val = extract_overlay_field("part of body affected", direction="below", max_dy=20, max_dx=200)
    if body_val:
        result["body_part"] = body_val

    # Injury description
    inj_val = extract_overlay_field("injury/illness that occurred", direction="below", max_dy=20, max_dx=250)
    if inj_val:
        result["injury_description"] = inj_val

    # Place of accident
    place_val = extract_overlay_field("place of accident", direction="below", max_dy=20, max_dx=400)
    if place_val:
        result["place_of_accident"] = place_val

    # --- Mark-mode circled-letter fields ---
    sex = extract_marked_letter(lines, marks, "sex", ["M", "F"], max_dy=25, max_dx=120)
    if sex:
        result["employee_sex"] = sex

    am_pm = extract_marked_letter(lines, marks, "time of accident", ["AM", "PM"], max_dy=25, max_dx=400)
    if am_pm:
        result["am_pm"] = am_pm

    if fitz_page is not None:
        try:
            doc.close()
        except Exception:
            pass

    return result


def structure_froi(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured fields from a FROI (First Report of Injury) form.

    Mode selection:
    - If the PDF contains non-black user marks (drawings in a distinctive
      color), use mark-mode extraction. This handles real-world DWC-1 forms
      with circled letters.
    - Otherwise use checkbox-based extraction for synthetic forms.

    The check runs on the PDF's drawings directly (not on already-detected
    `controls`) because real form templates often produce false-positive
    checkbox detections from their static vector shapes.
    """
    controls = page.get("controls", [])
    lines = page.get("lines", [])
    scale = 300 / 72

    # Real-form detection: if the PDF has any non-black marks, assume real form
    if pdf_path is not None:
        try:
            doc = fitz.open(pdf_path)
            marks = find_marks(doc[0])
            doc.close()
            if len(marks) > 0:
                return structure_froi_real(page, pdf_path)
        except Exception:
            pass

    result: dict[str, Any] = {"form_type": "froi", "extraction_mode": "checkboxes"}

    name = extract_employee_name(lines)
    if name:
        result["employee_name"] = name

    employer = extract_employer_name(lines)
    if employer:
        result["employer_name"] = employer

    inj_date = extract_injury_date(lines)
    if inj_date:
        result["injury_date"] = inj_date

    on_premises = extract_on_premises(lines, controls, scale)
    if on_premises is not None:
        result["on_premises"] = on_premises

    result["injured_body_parts"] = extract_injured_body_parts(lines, controls, scale)

    nature = extract_nature_of_injury(lines, controls, scale)
    if nature:
        result["nature_of_injury"] = nature

    result["causes"] = extract_causes(lines, controls, scale)
    result["treatments"] = extract_treatments(lines, controls, scale)

    # OCR/scan fallbacks using position-based mark detection
    is_ocr = page.get("text_source") == "ocr"
    if is_ocr and pdf_path:
        try:
            import fitz as _fitz
            from extracto.detection.controls import page_to_image as _pti
            from extracto.detection.position_mark import is_position_marked as _ipm
            _doc = _fitz.open(pdf_path)
            _img = _pti(_doc[0], dpi=300)
            _doc.close()

            def _pos_yn(anchor_text: str) -> bool | None:
                """Position-based Yes/No detection."""
                for ln in page.get("lines", []):
                    if anchor_text.lower() in ln["text"].lower():
                        by = (ln["bbox"][1] + ln["bbox"][3]) / 2
                        # Find Yes/No text on same row
                        yes_pos = no_pos = None
                        for ln2 in page.get("lines", []):
                            t = ln2["text"].strip().lower()
                            cy2 = (ln2["bbox"][1] + ln2["bbox"][3]) / 2
                            if abs(cy2 - by) > 20:
                                continue
                            if t in ("yes", "y") and yes_pos is None:
                                yes_pos = ((ln2["bbox"][0] + ln2["bbox"][2]) / 2, cy2)
                            elif t in ("no", "n") and no_pos is None:
                                no_pos = ((ln2["bbox"][0] + ln2["bbox"][2]) / 2, cy2)
                        if yes_pos and no_pos:
                            _, yd = _ipm(_img, yes_pos[0] - 10, yes_pos[1], size_pt=9.0, threshold=0.0)
                            _, nd = _ipm(_img, no_pos[0] - 10, no_pos[1], size_pt=9.0, threshold=0.0)
                            if max(yd, nd) >= 0.20:
                                if yd > nd * 1.15:
                                    return True
                                if nd > yd * 1.15:
                                    return False
                        break
                return None

            def _pos_multi(options: list[str], anchor: str | None = None) -> list[str]:
                """Position-based multi-select detection."""
                section_y = 0
                if anchor:
                    for ln in page.get("lines", []):
                        if anchor.lower() in ln["text"].lower():
                            section_y = ln["bbox"][1]
                            break
                selected = []
                for opt in options:
                    opt_l = opt.lower()
                    for ln in page.get("lines", []):
                        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
                        if cy < section_y:
                            continue
                        if opt_l in ln["text"].lower():
                            cx = ln["bbox"][0] - 8
                            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
                            marked, _ = _ipm(_img, cx, cy, size_pt=9.0, threshold=0.22)
                            if marked:
                                selected.append(opt)
                            break
                return selected

            # on_premises fallback
            if result.get("on_premises") is None:
                v = _pos_yn("on employer premises")
                if v is not None:
                    result["on_premises"] = v

            # nature_of_injury fallback
            if not result.get("nature_of_injury"):
                for nat in NATURE_OF_INJURY:
                    for ln in page.get("lines", []):
                        if nat.lower() in ln["text"].lower():
                            cx = ln["bbox"][0] - 8
                            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
                            marked, _ = _ipm(_img, cx, cy, size_pt=9.0, threshold=0.22)
                            if marked:
                                result["nature_of_injury"] = nat
                            break
                    if result.get("nature_of_injury"):
                        break

            # causes fallback
            if not result.get("causes"):
                pos_causes = _pos_multi(CAUSE_OF_INJURY, "cause of injury")
                if pos_causes:
                    result["causes"] = sorted(pos_causes)

            # treatments fallback
            if not result.get("treatments"):
                pos_treats = _pos_multi(TREATMENT_OPTIONS, "treatment received")
                if pos_treats:
                    result["treatments"] = sorted(pos_treats)

        except Exception:
            pass

    return result
