"""Parse HIPAA Authorization for Release of PHI forms.

Key structural feature: **opt-out checkbox semantics** for sensitive categories.
A checked box means "DO NOT release this category" — the opposite of usual
checkbox interpretation.

Extracted fields:
- patient_name, patient_dob (text fields)
- date_range_from, date_range_to (date range of records being released)
- record_types (set of record categories selected)
- excluded_categories (set of sensitive categories opted OUT - what was checked)
- purposes (set of disclosure purposes checked)
- expiration_type (one_year | specific_date)
"""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_CATEGORIES = [
    "Mental Health",
    "HIV/AIDS",
    "Substance Abuse",
    "Genetic Information",
]

PURPOSE_OPTIONS = [
    "Legal Proceedings",
    "Insurance Claim",
    "Personal Use",
    "Continuity of Care",
    "Other",
]

RECORD_TYPES = [
    "Office Notes",
    "Imaging Reports",
    "Lab Results",
    "Operative Reports",
    "Billing Records",
    "Discharge Summaries",
]

DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")


def _find_line_contains(lines, needle: str):
    """Multi-tier match: exact → whitespace-tolerant → fuzzy (for OCR'd text)."""
    from extracto.structuring.fuzzy import fuzzy_contains, normalize
    needle_norm = normalize(needle)
    needle_nospace = needle_norm.replace(" ", "")
    for ln in lines:
        text = normalize(ln["text"])
        if needle_norm in text or needle_nospace in text.replace(" ", ""):
            return ln
    # Fuzzy fallback for OCR garbling
    for ln in lines:
        if fuzzy_contains(ln["text"], needle, threshold=0.78):
            return ln
    return None


def _text_in_region(lines, x0, y0, x1, y1):
    hits = []
    for ln in lines:
        bbox = ln["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            hits.append(ln)
    return hits


def _nearest_text_right(lines, anchor_text: str, max_dx: float = 300, max_dy: float = 4) -> str | None:
    """Find the text that appears immediately to the right of a label like 'Patient Name:'."""
    anchor = _find_line_contains(lines, anchor_text)
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
            # Skip text that is itself another label (contains ':')
            if ":" in ln["text"] and ":" not in anchor_text:
                continue
            best = ln["text"].strip()
            best_dx = dx
    return best


def _find_checkbox_for_label(controls, lines, label_text: str, scale: float) -> dict | None:
    """Find the checkbox immediately to the left of a given label text."""
    label_line = None
    for ln in lines:
        if label_text.lower() in ln["text"].lower():
            label_line = ln
            break
    if not label_line:
        return None

    lx0, ly0, lx1, ly1 = label_line["bbox"]
    lcy = (ly0 + ly1) / 2

    # Find the closest checkbox to the left of the label text on the same y row
    best = None
    best_dx = 1e9
    for c in controls:
        if c["kind"] != "checkbox":
            continue
        x, y, w, h = c["bbox"]
        cy = (y + h / 2) / scale
        cx_right = (x + w) / scale
        if abs(cy - lcy) > 8:
            continue
        dx = lx0 - cx_right
        if 0 <= dx <= 15 and dx < best_dx:
            best = c
            best_dx = dx
    return best


def extract_patient_name(lines) -> str | None:
    val = _nearest_text_right(lines, "patient name", max_dx=350, max_dy=5)
    if val:
        return val
    return _nearest_text_right(lines, "patient's name", max_dx=350, max_dy=5)


def extract_patient_dob(lines) -> str | None:
    val = _nearest_text_right(lines, "date of birth", max_dx=250, max_dy=5)
    if val and DATE_RE.search(val):
        return DATE_RE.search(val).group(1)
    return None


def extract_date_range(lines) -> tuple[str | None, str | None]:
    """Extract the 'From' and 'To' service dates."""
    anchor = _find_line_contains(lines, "date of service from")
    if not anchor:
        anchor = _find_line_contains(lines, "date of service")
    if not anchor:
        return None, None
    ax0, ay0, ax1, ay1 = anchor["bbox"]
    band_cy = (ay0 + ay1) / 2

    # Collect dates on the same row
    dates: list[tuple[float, str]] = []
    for ln in lines:
        bbox = ln["bbox"]
        lcy = (bbox[1] + bbox[3]) / 2
        if abs(lcy - band_cy) > 5:
            continue
        for m in DATE_RE.finditer(ln["text"]):
            dates.append((bbox[0], m.group(1)))

    dates.sort(key=lambda d: d[0])
    if len(dates) >= 2:
        return dates[0][1], dates[1][1]
    if len(dates) == 1:
        return dates[0][1], None
    return None, None


def extract_record_types(lines, controls, scale) -> list[str]:
    """Find which record types are checked in the 'Records to Be Released' section."""
    selected: list[str] = []
    for rec_type in RECORD_TYPES:
        box = _find_checkbox_for_label(controls, lines, rec_type, scale)
        if box and box["selected"]:
            selected.append(rec_type)
    return sorted(selected)


def extract_excluded_categories(lines, controls, scale) -> list[str]:
    """Find which sensitive categories are opted OUT.

    In HIPAA forms, the checkbox labels are 'Do NOT release <Category>'. A
    checked box means that category is EXCLUDED from the release.
    """
    excluded: list[str] = []
    for category in SENSITIVE_CATEGORIES:
        # The label text is 'Do NOT release <category>' - find that line
        label_phrase = f"do not release {category.lower()}"
        anchor = None
        for ln in lines:
            if label_phrase in ln["text"].lower():
                anchor = ln
                break
        if not anchor:
            # Fallback: just match the category name itself within the sensitive section
            for ln in lines:
                if category.lower() in ln["text"].lower() and "do not release" in ln["text"].lower():
                    anchor = ln
                    break
        if not anchor:
            continue

        lx0, ly0, lx1, ly1 = anchor["bbox"]
        lcy = (ly0 + ly1) / 2

        # Find the checkbox to the left
        for c in controls:
            if c["kind"] != "checkbox":
                continue
            x, y, w, h = c["bbox"]
            cy = (y + h / 2) / scale
            cx_right = (x + w) / scale
            if abs(cy - lcy) > 8:
                continue
            dx = lx0 - cx_right
            if 0 <= dx <= 20:
                if c["selected"]:
                    excluded.append(category)
                break

    return sorted(excluded)


def extract_purposes(lines, controls, scale) -> list[str]:
    """Find which disclosure purposes are checked."""
    # Constrain to the 'PURPOSE OF DISCLOSURE' section to avoid matching
    # 'Legal Proceedings' text elsewhere.
    section_anchor = _find_line_contains(lines, "purpose of disclosure")
    if not section_anchor:
        return []
    sy1 = section_anchor["bbox"][3]

    # The purposes section typically spans ~50pt below the header
    purpose_region_y0 = sy1
    purpose_region_y1 = sy1 + 60

    selected: list[str] = []
    for purpose in PURPOSE_OPTIONS:
        # Find a line containing this purpose name within the region
        match = None
        for ln in lines:
            if purpose.lower() not in ln["text"].lower():
                continue
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            if purpose_region_y0 <= cy <= purpose_region_y1:
                match = ln
                break
        if not match:
            continue

        lx0, ly0, lx1, ly1 = match["bbox"]
        lcy = (ly0 + ly1) / 2
        for c in controls:
            if c["kind"] != "checkbox":
                continue
            x, y, w, h = c["bbox"]
            cy = (y + h / 2) / scale
            cx_right = (x + w) / scale
            if abs(cy - lcy) > 8:
                continue
            dx = lx0 - cx_right
            if 0 <= dx <= 20:
                if c["selected"]:
                    selected.append(purpose)
                break
    return sorted(selected)


def extract_expiration_type(lines, controls, scale) -> str | None:
    """Check which expiration option is selected: one_year or specific_date."""
    one_year_anchor = _find_line_contains(lines, "one year from date")
    specific_anchor = _find_line_contains(lines, "specific date")

    one_year_selected = False
    specific_selected = False

    for anchor, name in [(one_year_anchor, "one_year"), (specific_anchor, "specific")]:
        if not anchor:
            continue
        lx0, ly0, lx1, ly1 = anchor["bbox"]
        lcy = (ly0 + ly1) / 2
        for c in controls:
            if c["kind"] != "checkbox":
                continue
            x, y, w, h = c["bbox"]
            cy = (y + h / 2) / scale
            cx_right = (x + w) / scale
            if abs(cy - lcy) > 8:
                continue
            dx = lx0 - cx_right
            if 0 <= dx <= 20:
                if c["selected"]:
                    if name == "one_year":
                        one_year_selected = True
                    else:
                        specific_selected = True
                break

    if one_year_selected and not specific_selected:
        return "one_year"
    if specific_selected and not one_year_selected:
        return "specific_date"
    return None


def _position_mark_multi_select(
    page: dict[str, Any], pdf_path: str | None,
    options: list[str],
    section_anchor: str | None = None,
) -> list[str] | None:
    """Position-based multi-select for scanned forms.

    Finds each option text in OCR lines, checks for a mark to the left.
    Returns list of selected options.
    """
    if not pdf_path or page.get("text_source") != "ocr":
        return None
    try:
        import fitz
        from extracto.detection.controls import page_to_image
        from extracto.detection.position_mark import is_position_marked
        doc = fitz.open(pdf_path)
        img = page_to_image(doc[0], dpi=300)
        doc.close()
    except Exception:
        return None

    lines = page.get("lines", [])

    # Find section anchor y to constrain search
    section_y = 0
    if section_anchor:
        for ln in lines:
            if section_anchor.lower() in ln["text"].lower():
                section_y = ln["bbox"][1]
                break

    selected: list[str] = []
    for opt in options:
        opt_lower = opt.lower()
        for ln in lines:
            text = ln["text"].strip().lower()
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            if cy < section_y:
                continue
            if opt_lower in text or text in opt_lower:
                bbox = ln["bbox"]
                # Check for mark to the LEFT of the option text
                cx = bbox[0] - 8
                cy = (bbox[1] + bbox[3]) / 2
                marked, density = is_position_marked(img, cx, cy, size_pt=9.0, threshold=0.22)
                if marked:
                    selected.append(opt)
                break

    return selected if selected else None


def structure_hipaa(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured fields from a HIPAA Authorization form."""
    controls = page.get("controls", [])
    lines = page.get("lines", [])
    scale = 300 / 72
    is_ocr = page.get("text_source") == "ocr"

    result: dict[str, Any] = {"form_type": "hipaa"}

    name = extract_patient_name(lines)
    if name:
        result["patient_name"] = name

    dob = extract_patient_dob(lines)
    if dob:
        result["patient_dob"] = dob

    d_from, d_to = extract_date_range(lines)
    if d_from:
        result["date_range_from"] = d_from
    if d_to:
        result["date_range_to"] = d_to

    result["record_types"] = extract_record_types(lines, controls, scale)
    result["excluded_categories"] = extract_excluded_categories(lines, controls, scale)
    result["purposes"] = extract_purposes(lines, controls, scale)

    exp = extract_expiration_type(lines, controls, scale)
    if exp:
        result["expiration_type"] = exp

    # OCR/scan fallbacks using position-based mark detection
    if is_ocr and pdf_path:
        # Record types fallback
        if not result.get("record_types") or (isinstance(result.get("record_types"), list) and not result["record_types"]):
            pos_records = _position_mark_multi_select(
                page, pdf_path, RECORD_TYPES, section_anchor="records to be released"
            )
            if pos_records:
                result["record_types"] = sorted(pos_records)

        # Purposes fallback
        if not result.get("purposes") or (isinstance(result.get("purposes"), list) and not result["purposes"]):
            pos_purposes = _position_mark_multi_select(
                page, pdf_path, PURPOSE_OPTIONS, section_anchor="purpose of disclosure"
            )
            if pos_purposes:
                result["purposes"] = sorted(pos_purposes)

        # Excluded categories fallback
        if not result.get("excluded_categories") or (isinstance(result.get("excluded_categories"), list) and not result["excluded_categories"]):
            pos_excluded = _position_mark_multi_select(
                page, pdf_path,
                [f"Do NOT release {cat}" for cat in SENSITIVE_CATEGORIES],
                section_anchor="specially protected"
            )
            if pos_excluded:
                # Map back from "Do NOT release X" to just "X"
                result["excluded_categories"] = sorted([
                    cat for cat in SENSITIVE_CATEGORIES
                    if f"Do NOT release {cat}" in pos_excluded
                ])

        # Expiration type fallback
        if not result.get("expiration_type"):
            pos_exp = _position_mark_multi_select(
                page, pdf_path,
                ["One year from date of signature", "Specific date"],
                section_anchor="expiration"
            )
            if pos_exp:
                if any("one year" in e.lower() for e in pos_exp):
                    result["expiration_type"] = "one_year"
                elif any("specific" in e.lower() for e in pos_exp):
                    result["expiration_type"] = "specific_date"

    return result
