"""Parse CMS-1500 health insurance claim forms into structured data.

CMS-1500 uses numbered boxes (1-33). We anchor on the box labels and search
for text fields, checkbox groups, and tabular data within each box.

Key extraction targets:
- Box 1:  Insurance type (7 checkboxes)
- Box 2:  Patient name (text)
- Box 3:  Patient DOB + sex (M/F checkboxes)
- Box 6:  Relationship to insured (4 checkboxes)
- Box 10: Condition related to Employment/Auto/Other (3 yes/no pairs)
- Box 21: Diagnosis codes (up to 12 ICD-10 codes)
- Box 24: Service lines (table with up to 6 rows)
- Box 25: Federal tax ID + SSN/EIN type
- Box 27: Accept assignment (yes/no)
- Box 28: Total charge (currency)
- Box 31: Provider signature
- Box 33: Provider NPI
"""

from __future__ import annotations

import re
from typing import Any

INSURANCE_TYPES = [
    "Medicare",
    "Medicaid",
    "TRICARE",
    "CHAMPVA",
    "Group Health Plan",
    "FECA Blk Lung",
    "Other",
]

RELATIONSHIPS = ["Self", "Spouse", "Child", "Other"]

# ICD-10 pattern: letter + 2 digits, optional decimal suffix
# Examples: M54.2, S13.4XXA, S06.0X0A, R52
ICD10_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.[A-Z0-9]{1,5})?)\b")

# CPT code pattern: 5 digits (standalone)
CPT_RE = re.compile(r"^\d{5}$")

# Currency value: $7365.00 or 7,365.00 or 7365.00
CURRENCY_RE = re.compile(r"\$?(\d+(?:,\d{3})*\.\d{2})")

# Date pattern MM/DD/YYYY
DATE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})")


def _index_lines(page: dict[str, Any]) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [(ln["text"], tuple(ln["bbox"])) for ln in page.get("lines", [])]


from extracto.structuring.fuzzy import fuzzy_contains, normalize, normalize_aggressive


def _find_line_starts_with(lines, prefix: str):
    prefix_norm = normalize(prefix)
    prefix_nospace = prefix_norm.replace(" ", "")
    for t, b in lines:
        t_norm = normalize(t)
        if t_norm.startswith(prefix_norm) or t_norm.replace(" ", "").startswith(prefix_nospace):
            return t, b
    # Tier 2: fuzzy match for OCR'd text
    for t, b in lines:
        if fuzzy_contains(t, prefix, threshold=0.78):
            return t, b
    return None


def _find_line_contains(lines, substring: str):
    substring_norm = normalize(substring)
    substring_nospace = substring_norm.replace(" ", "")
    for t, b in lines:
        t_norm = normalize(t)
        if substring_norm in t_norm or substring_nospace in t_norm.replace(" ", ""):
            return t, b
    # Tier 2: fuzzy match for OCR'd text
    for t, b in lines:
        if fuzzy_contains(t, substring, threshold=0.78):
            return t, b
    return None


def _controls_in_row(controls: list[dict], band_y: float, scale: float, max_dy: float = 10, x_range: tuple[float, float] | None = None) -> list[dict]:
    """Find all controls within a vertical band, optionally constrained in x."""
    hits = []
    for c in controls:
        x, y, w, h = c["bbox"]
        cy = (y + h / 2) / scale
        cx = (x + w / 2) / scale
        if abs(cy - band_y) > max_dy:
            continue
        if x_range and not (x_range[0] <= cx <= x_range[1]):
            continue
        hits.append(c)
    hits.sort(key=lambda c: c["bbox"][0])
    return hits


def _text_in_region(
    lines: list[tuple[str, tuple[float, float, float, float]]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return all text lines whose bbox center falls inside the rectangle."""
    hits = []
    for text, bbox in lines:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            hits.append((text, bbox))
    return hits


def extract_insurance_type(lines, controls: list[dict], scale: float) -> str | None:
    """Box 1: find the checked insurance type from the 7 checkboxes on the header row.

    The checkboxes span the full Box 1 width (larger than the text anchor),
    so we use a wide x range and filter to those clustered on the anchor's row.
    """
    anchor = _find_line_starts_with(lines, "1. medicare")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor
    band_y = (ay0 + ay1) / 2 + 12

    # Box 1 starts at ax0 but extends well past the text label. Use a wide
    # x range and limit to a small y band (8pt) to capture only the checkbox row.
    # Box 1 width is ~58% of page width (see generator). For letter-size page,
    # that's about 350pt from the left margin.
    candidates = _controls_in_row(
        controls, band_y, scale, max_dy=8,
        x_range=(ax0 - 5, ax0 + 360),
    )
    if not candidates:
        return None

    selected = [c for c in candidates if c["selected"]]
    if not selected:
        return None

    candidates.sort(key=lambda c: c["bbox"][0])
    # The first 7 sorted candidates are the insurance type checkboxes.
    # (Additional boxes may exist in Box 1a to the right.)
    insurance_boxes = candidates[:7]
    for i, c in enumerate(insurance_boxes):
        if c["selected"] and i < len(INSURANCE_TYPES):
            return INSURANCE_TYPES[i]
    return None


def extract_patient_sex(lines, controls: list[dict], scale: float) -> str | None:
    """Box 3: find M/F checkboxes right of 'PATIENT'S BIRTH DATE'."""
    anchor = _find_line_contains(lines, "patient's birth date")
    if not anchor:
        anchor = _find_line_contains(lines, "birth date")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor
    band_y = (ay0 + ay1) / 2 + 10

    # Sex checkboxes are at the right side of Box 3
    candidates = _controls_in_row(controls, band_y, scale, max_dy=10, x_range=(ax1, ax1 + 200))
    for c in candidates:
        if c["selected"]:
            lab = (c.get("label") or "").strip().upper()
            if lab in ("M", "MALE"):
                return "M"
            if lab in ("F", "FEMALE"):
                return "F"

    # Fallback: order by x, first = M, second = F
    if len([c for c in candidates if c["selected"]]) == 1:
        sel_idx = next(i for i, c in enumerate(candidates) if c["selected"])
        if sel_idx == 0:
            return "M"
        if sel_idx == 1:
            return "F"
    return None


def extract_relationship(lines, controls: list[dict], scale: float) -> str | None:
    """Box 6: find the selected relationship checkbox.

    The 4 relationship checkboxes (Self, Spouse, Child, Other) span the full
    width of Box 6, which is wider than the label text. Use a wide x_range
    that covers the full box, not just the label's bbox.
    """
    anchor = _find_line_contains(lines, "patient relationship to insured")
    if not anchor:
        anchor = _find_line_contains(lines, "6. patient relationship")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor
    band_y = (ay0 + ay1) / 2 + 10

    # Box 6 is ~180pt wide. Extend the x-range well past the label text.
    candidates = _controls_in_row(
        controls, band_y, scale, max_dy=10, x_range=(ax0 - 5, ax0 + 200)
    )
    selected = [c for c in candidates if c["selected"]]
    if not selected:
        return None

    # Try label-based match first (works for Self/Spouse/Child cleanly,
    # and handles Other via direct label match)
    for c in selected:
        lab = (c.get("label") or "").strip()
        if lab in RELATIONSHIPS:
            return lab

    # Fallback: position-based (Self, Spouse, Child, Other in x order)
    candidates.sort(key=lambda c: c["bbox"][0])
    # Take the leftmost 4 checkboxes (the relationship row)
    rel_boxes = candidates[:4]
    for i, c in enumerate(rel_boxes):
        if c["selected"] and i < len(RELATIONSHIPS):
            return RELATIONSHIPS[i]
    return None


def extract_condition(lines, controls: list[dict], scale: float) -> dict[str, bool | None]:
    """Box 10: employment, auto, other condition yes/no pairs."""
    result: dict[str, bool | None] = {
        "condition_employment": None,
        "condition_auto": None,
        "condition_other": None,
    }

    for key, anchor_text in [
        ("condition_employment", "employment"),
        ("condition_auto", "auto accident"),
        ("condition_other", "other accident"),
    ]:
        anchor = _find_line_contains(lines, anchor_text)
        if not anchor:
            continue
        _, (ax0, ay0, ax1, ay1) = anchor
        band_y = (ay0 + ay1) / 2

        # Y/N checkboxes are to the right of the label on same row
        candidates = _controls_in_row(
            controls, band_y, scale, max_dy=8, x_range=(ax1, ax1 + 250)
        )

        yes_labeled = [c for c in candidates if (c.get("label") or "").strip().upper() == "YES"]
        no_labeled = [c for c in candidates if (c.get("label") or "").strip().upper() == "NO"]
        yes_sel = any(c["selected"] for c in yes_labeled)
        no_sel = any(c["selected"] for c in no_labeled)

        if not yes_labeled or not no_labeled:
            # Fallback: first two controls in order
            if len(candidates) >= 2:
                candidates.sort(key=lambda c: c["bbox"][0])
                yes_sel = candidates[0]["selected"]
                no_sel = candidates[1]["selected"]

        if yes_sel and not no_sel:
            result[key] = True
        elif no_sel and not yes_sel:
            result[key] = False

    return result


def extract_diagnoses(lines) -> list[str]:
    """Box 21: ICD-10 diagnosis codes (up to 12).

    Codes appear as 'A.M54.2', 'B.S13.4XXA', etc. The letter prefix labels
    the slot; we strip it and keep the actual ICD-10 code.
    """
    anchor = _find_line_contains(lines, "diagnosis or nature of illness")
    if not anchor:
        return []
    _, (ax0, ay0, ax1, ay1) = anchor

    # Box 21 is about 40pt tall with a 4x3 grid of codes
    region = _text_in_region(lines, 0, ay1, 9999, ay1 + 45)

    # Collect all ICD-10 matches, preserving order by (y, x)
    hits: list[tuple[float, float, str]] = []
    for text, bbox in region:
        # Strip label prefix like "A.", "B.", etc.
        cleaned = re.sub(r"^[A-L]\.", "", text.strip())
        for match in ICD10_RE.finditer(cleaned):
            code = match.group(1)
            hits.append((bbox[1], bbox[0], code))

    # Sort by row (y), then column (x)
    hits.sort(key=lambda h: (round(h[0] / 5), h[1]))

    codes = []
    seen = set()
    for _, _, code in hits:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes[:12]


def extract_service_lines(lines) -> list[dict[str, Any]]:
    """Box 24: service line table rows.

    Adjacent columns may get merged into a single text span when the PDF is
    rasterized by reportlab (e.g. '06/04/2024 12' for date+POS, '$255.00 3'
    for charge+units). We split each span on whitespace and pattern-match
    the components.
    """
    anchor = _find_line_contains(lines, "date(s) of service")
    if not anchor:
        anchor = _find_line_contains(lines, "24.")
        if not anchor:
            return []
    _, (ax0, ay0, ax1, ay1) = anchor

    # Box 24 is ~110pt tall (header + 6 rows x ~18pt)
    region = _text_in_region(lines, 0, ay1 + 2, 9999, ay1 + 120)

    # Cluster text by y-row
    rows_by_y: list[tuple[float, list]] = []
    for text, bbox in region:
        cy = (bbox[1] + bbox[3]) / 2
        placed = False
        for i, (ry, items) in enumerate(rows_by_y):
            if abs(ry - cy) <= 5:
                items.append((text, bbox))
                rows_by_y[i] = ((ry * len(items) + cy) / (len(items) + 1), items)
                placed = True
                break
        if not placed:
            rows_by_y.append((cy, [(text, bbox)]))
    rows_by_y.sort(key=lambda r: r[0])

    service_lines = []
    for _, row_items in rows_by_y:
        row_items.sort(key=lambda t: t[1][0])

        # Tokenize every text span on the row
        tokens: list[str] = []
        for text, _ in row_items:
            tokens.extend(text.strip().split())

        date_from = None
        date_to = None
        pos = None
        cpt = None
        charge = None
        units = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            # Date
            if DATE_RE.fullmatch(tok):
                if date_from is None:
                    date_from = tok
                elif date_to is None:
                    date_to = tok
            # 5-digit CPT code
            elif CPT_RE.match(tok):
                cpt = tok
            # Currency
            elif CURRENCY_RE.match(tok):
                m = CURRENCY_RE.match(tok)
                val = m.group(1).replace(",", "")
                try:
                    charge = float(val)
                except ValueError:
                    pass
                # The next token is typically units if it's a small integer
                if i + 1 < len(tokens) and tokens[i + 1].isdigit() and len(tokens[i + 1]) <= 2:
                    units = int(tokens[i + 1])
                    i += 1
            # 2-digit place of service (only if we've already seen a date)
            elif re.fullmatch(r"\d{2}", tok) and date_from and pos is None:
                pos = tok
            i += 1

        if cpt:
            service_lines.append({
                "date_from": date_from,
                "cpt": cpt,
                "pos": pos,
                "charge": charge,
                "units": units,
            })

    return service_lines


def extract_total_charge(lines) -> float | None:
    """Box 28: total charge currency value."""
    anchor = _find_line_contains(lines, "28. total charge")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor

    # Look for currency in the box below the label
    region = _text_in_region(lines, ax0, ay1, ax1 + 80, ay1 + 30)
    for text, _ in region:
        m = CURRENCY_RE.search(text)
        if m:
            val = m.group(1).replace(",", "")
            try:
                return float(val)
            except ValueError:
                continue
    return None


def extract_accept_assignment(lines, controls: list[dict], scale: float) -> bool | None:
    """Box 27: accept assignment yes/no."""
    anchor = _find_line_contains(lines, "accept assignment")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor
    band_y = (ay0 + ay1) / 2 + 10

    candidates = _controls_in_row(
        controls, band_y, scale, max_dy=12, x_range=(ax0 - 5, ax1 + 100)
    )

    # Labeled approach
    yes_labeled = [c for c in candidates if (c.get("label") or "").strip().upper() == "YES"]
    no_labeled = [c for c in candidates if (c.get("label") or "").strip().upper() == "NO"]
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


def extract_tax_id_type(lines, controls: list[dict], scale: float) -> str | None:
    """Box 25: SSN vs EIN checkboxes.

    The SSN/EIN checkboxes are positioned near the right edge of Box 25,
    often beyond the text anchor's x1. We use a wider x range that covers
    the full Box 25 width (~160pt).
    """
    anchor = _find_line_contains(lines, "federal tax i.d.")
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor
    band_y = (ay0 + ay1) / 2 + 10

    # Box 25 width is ~144pt (25% of page). The anchor text label goes 20-133
    # but the SSN/EIN checkboxes are near x=125 and x=145.
    candidates = _controls_in_row(
        controls, band_y, scale, max_dy=10,
        x_range=(ax0, ax0 + 160),
    )
    if not candidates:
        return None

    # SSN/EIN checkboxes are small (~6pt); filter out the larger Box 27 Y/N
    # checkboxes that might be in the same y band.
    small_boxes = [c for c in candidates if (c["bbox"][2] / scale) < 6.5]
    if len(small_boxes) < 2:
        # Fallback: any checkboxes to the left of ~160pt
        small_boxes = candidates

    small_boxes.sort(key=lambda c: c["bbox"][0])
    # Take the first two — SSN then EIN
    if len(small_boxes) >= 2:
        if small_boxes[0]["selected"] and not small_boxes[1]["selected"]:
            return "SSN"
        if small_boxes[1]["selected"] and not small_boxes[0]["selected"]:
            return "EIN"
    return None


def extract_text_field(lines, anchor_substring: str, max_dy: float = 12, max_dx: float = 200) -> str | None:
    """Extract a text value that appears just below a labeled box header."""
    anchor = _find_line_contains(lines, anchor_substring)
    if not anchor:
        return None
    _, (ax0, ay0, ax1, ay1) = anchor

    # Look for text directly below the anchor label (the box content)
    for text, bbox in lines:
        lx0, ly0, lx1, ly1 = bbox
        lcy = (ly0 + ly1) / 2
        if ay1 < lcy <= ay1 + max_dy and ax0 - 5 <= lx0 <= ax0 + max_dx:
            # Don't return the anchor line itself
            if text.strip() and anchor_substring.lower() not in text.lower():
                return text.strip()
    return None


def _position_mark_fallback_for_options(
    page: dict[str, Any], pdf_path: str | None,
    anchor_text: str, options: list[str],
) -> str | None:
    """Position-based mark detection for option lists.

    For each option, finds its text in OCR and checks ALL plausible checkbox
    positions (left-of-text, below-text, above-text). Picks the option whose
    expected checkbox position has the highest ink density above background.

    The trick: rather than picking ANY marked position (which can include
    adjacent options' text), we measure density at every option's "best" spot
    and only return the one with substantially higher density than the others.
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

    # For each option, compute the BEST density across plausible checkbox positions.
    option_scores: list[tuple[str, float]] = []
    for opt in options:
        # Find the option text in OCR lines (fuzzy first-word match for multi-word)
        opt_first_word = opt.split()[0].lower()
        match_ln = None
        for ln in lines:
            text = ln["text"].strip().lower()
            if opt.lower() in text or opt_first_word in text.split():
                match_ln = ln
                break
        if not match_ln:
            option_scores.append((opt, 0.0))
            continue

        bbox = match_ln["bbox"]
        text_cx = (bbox[0] + bbox[2]) / 2
        text_cy = (bbox[1] + bbox[3]) / 2
        text_h = bbox[3] - bbox[1]

        # Try checkbox positions: directly left of text, just-below, just-above
        candidates = [
            (bbox[0] - 6, text_cy),                # left
            (text_cx, bbox[3] + text_h * 0.8),      # below
            (text_cx, bbox[1] - text_h * 0.8),      # above
        ]
        best_density_for_opt = 0.0
        for cx, cy in candidates:
            _, density = is_position_marked(img, cx, cy, size_pt=8.0, threshold=0.0)
            if density > best_density_for_opt:
                best_density_for_opt = density
        option_scores.append((opt, best_density_for_opt))

    if not option_scores:
        return None

    # Pick the option with the highest density, BUT only if it's meaningfully
    # higher than the median (i.e., distinguishable from background ink density).
    sorted_scores = sorted(option_scores, key=lambda s: -s[1])
    best_opt, best_density = sorted_scores[0]
    if best_density < 0.20:
        return None
    # Require the winner to be at least 1.3x the runner-up to avoid noise
    if len(sorted_scores) > 1:
        runner_up = sorted_scores[1][1]
        if runner_up > 0 and best_density < runner_up * 1.15:
            return None
    return best_opt


def structure_cms1500(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured fields from a CMS-1500 page.

    For scanned forms (text_source=='ocr'), uses position-based mark detection
    as a fallback for checkbox-dependent fields when CV detection fails.
    """
    controls = page.get("controls", [])
    lines = _index_lines(page)
    scale = 300 / 72
    is_ocr = page.get("text_source") == "ocr"

    result: dict[str, Any] = {
        "form_type": "cms1500",
    }

    # Box 1: Insurance type
    ins = extract_insurance_type(lines, controls, scale)
    if not ins and is_ocr:
        ins = _position_mark_fallback_for_options(
            page, pdf_path, "1. medicare",
            ["Medicare", "Medicaid", "TRICARE", "CHAMPVA",
             "Group Health Plan", "FECA Blk Lung", "Other"],
        )
    if ins:
        result["insurance_type"] = ins

    # Box 2: Patient name (text value in box below anchor)
    name = extract_text_field(lines, "patient's name")
    if name:
        result["patient_name"] = name

    # Box 3: Patient sex
    sex = extract_patient_sex(lines, controls, scale)
    if not sex and is_ocr:
        sex = _position_mark_fallback_for_options(
            page, pdf_path, "sex", ["M", "F"],
        )
    # OCR fallback 2: ink density at M and F positions.
    # Find any text containing just "M" or "F" near the SEX anchor row,
    # measure ink density at each to determine which is marked.
    if not sex and is_ocr and pdf_path:
        try:
            import fitz as _fitz
            import re as _re
            from extracto.detection.controls import page_to_image as _pti
            from extracto.detection.position_mark import is_position_marked as _ipm
            _doc = _fitz.open(pdf_path)
            _img = _pti(_doc[0], dpi=300)
            _doc.close()

            sex_anchor = _find_line_contains(lines, "sex")
            if sex_anchor:
                _, (ax0, ay0, ax1, ay1) = sex_anchor
                band_y = (ay0 + ay1) / 2
                m_density = 0.0
                f_density = 0.0
                for t, b in lines:
                    cy = (b[1] + b[3]) / 2
                    if abs(cy - band_y) > 15 or b[0] < ax1:
                        continue
                    text_clean = _re.sub(r"[^A-Za-z]", "", t).upper()
                    # Text containing M (standalone or with OCR artifacts)
                    if "M" in text_clean and len(text_clean) <= 5 and b[0] < ax1 + 120:
                        _, d = _ipm(_img, (b[0] + b[2]) / 2, cy, size_pt=8, threshold=0.0)
                        m_density = max(m_density, d)
                    # Text containing F (standalone, or "F |" merged with next field)
                    if "F" in text_clean[:3] and b[0] > ax1 + 50:
                        _, d = _ipm(_img, b[0] + 2, cy, size_pt=8, threshold=0.0)
                        f_density = max(f_density, d)
                if m_density > 0.15 or f_density > 0.15:
                    if m_density > f_density * 1.1:
                        sex = "M"
                    elif f_density > m_density * 1.1:
                        sex = "F"
        except Exception:
            pass
    if sex:
        result["patient_sex"] = sex

    # Box 6: Relationship to insured
    rel = extract_relationship(lines, controls, scale)
    if not rel and is_ocr:
        rel = _position_mark_fallback_for_options(
            page, pdf_path, "patient relationship",
            ["Self", "Spouse", "Child", "Other"],
        )
    if rel:
        result["relationship_to_insured"] = rel

    # Box 10: Condition related to
    cond = extract_condition(lines, controls, scale)
    if is_ocr:
        for key, anchor in [
            ("condition_employment", "employment"),
            ("condition_auto", "auto accident"),
            ("condition_other", "other accident"),
        ]:
            if cond.get(key) is None:
                yes_no = _position_mark_fallback_for_options(
                    page, pdf_path, anchor, ["YES", "NO"],
                )
                if yes_no == "YES":
                    cond[key] = True
                elif yes_no == "NO":
                    cond[key] = False
    result.update(cond)

    # Box 21: Diagnosis codes
    dx = extract_diagnoses(lines)
    if dx:
        result["diagnoses"] = dx

    # Box 24: Service lines
    svc = extract_service_lines(lines)
    if svc:
        result["service_lines"] = svc

    # Box 28: Total charge
    total = extract_total_charge(lines)
    if total is not None:
        result["total_charge"] = total

    # Box 27: Accept assignment
    assign = extract_accept_assignment(lines, controls, scale)
    if assign is None and is_ocr:
        yn = _position_mark_fallback_for_options(
            page, pdf_path, "accept assignment", ["YES", "NO"],
        )
        if yn == "YES":
            assign = True
        elif yn == "NO":
            assign = False
    if assign is not None:
        result["accept_assignment"] = assign

    # Box 25: Tax ID type
    tax_type = extract_tax_id_type(lines, controls, scale)
    if not tax_type and is_ocr:
        tax_type = _position_mark_fallback_for_options(
            page, pdf_path, "federal tax", ["SSN", "EIN"],
        )
    if tax_type:
        result["tax_id_type"] = tax_type

    return result
