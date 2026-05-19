"""Map detected controls to semantic form fields using anchor-based spatial reasoning.

Finds text anchors (e.g. "Sex:", "Smoker", "Allergies") then searches for controls
in a spatial band relative to the anchor position.

Handles multiple form layout variants (v1: 2-column with verbose labels,
v2: 3-column with terse labels).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable

ALLERGY_OPTIONS = ["Penicillin", "Peanuts", "Latex", "Shellfish", "Pollen", "Dust", "None Known"]
SYMPTOMS = ["Fever", "Cough", "Headache", "Fatigue", "Shortness of breath", "Nausea", "Dizziness", "Chest pain"]
CLAIM_OPTS = ["Visit", "Procedure", "Medication", "Other"]

# Anchor text variants to support multiple form layouts
SEX_ANCHORS = ("sex", "gender")
WORK_RELATED_ANCHORS = ("is this work-related", "work-related", "work related")
AUTO_ACCIDENT_ANCHORS = ("auto accident", "motor vehicle", "mva")
SYMPTOMS_ANCHORS = ("current symptoms", "symptoms", "presenting symptoms", "symptom")
ALLERGIES_ANCHORS = ("allergies", "allergy", "known allergies", "check all that apply", "check all")


def _token_set_ratio(a: str, b: str) -> float:
    """Blended Jaccard + SequenceMatcher similarity for fuzzy matching."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    jacc = len(ta & tb) / len(ta | tb)
    sm = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.6 * jacc + 0.4 * sm


def _index_by_text(lines: list[dict[str, Any]]) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [(ln["text"], tuple(ln["bbox"])) for ln in lines]


def _find_line(lines, predicate: Callable[[str], bool]):
    """Find first line matching predicate."""
    for t, b in lines:
        if predicate(t):
            return t, b
    return None


def _find_anchor_by_variants(lines, variants: tuple[str, ...]):
    """Find first line matching any anchor variant, preferring longer matches first.

    Three-tier match: exact substring → whitespace-tolerant → fuzzy (Levenshtein).
    The fuzzy tier handles OCR garbling like 'Fatigue' → 'Gighttatique'.
    """
    from extracto.structuring.fuzzy import fuzzy_contains
    ordered = sorted(variants, key=len, reverse=True)
    # Tier 1: exact lowercase substring
    for variant in ordered:
        hit = _find_line(lines, lambda t, v=variant: v in t.lower())
        if hit:
            return hit
    # Tier 2: fuzzy (handles OCR errors)
    for variant in ordered:
        if len(variant) < 4:
            continue
        hit = _find_line(lines, lambda t, v=variant: fuzzy_contains(t, v, threshold=0.78))
        if hit:
            return hit
    return None


def _clean_ocr_text(text: str) -> str:
    """Strip common OCR artifacts where checkbox/radio circles get read as chars.

    Examples: "© Male" → "Male", "@) Female" → "Female", "So. Yes" → "Yes",
    "O Yes" → "Yes", "[X] No" → "No"
    """
    import re
    # Strip leading symbols/brackets that OCR reads from checkbox/radio shapes
    cleaned = re.sub(r"^[\[(\]©®@O□■☐☑✓✗Xx\s.)\]]+", "", text).strip()
    # Also handle "So. Yes" → strip leading words that aren't Yes/No/Male/Female
    words = cleaned.split()
    if len(words) >= 2 and words[-1].lower() in ("yes", "no", "male", "female", "other"):
        cleaned = words[-1]
    return cleaned


def _ocr_position_yes_no(
    page: dict[str, Any], pdf_path: str | None,
    anchor_substring: str,
) -> bool | None:
    """Position-based Yes/No detection for scanned forms.

    Finds the anchor text, then checks for marks at expected Yes/No positions.
    Handles OCR garbling like '© No', 'So. Yes', '@) Female'.
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
    anchor_y = None
    for ln in lines:
        if anchor_substring.lower() in ln["text"].lower():
            anchor_y = (ln["bbox"][1] + ln["bbox"][3]) / 2
            break
    if anchor_y is None:
        return None

    # Find Yes and No positions — with OCR artifact cleanup
    yes_pos = None
    no_pos = None
    for ln in lines:
        raw_text = ln["text"].strip()
        cleaned = _clean_ocr_text(raw_text).lower()
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        if abs(cy - anchor_y) > 30:
            continue
        if cleaned in ("yes", "y") and yes_pos is None:
            yes_pos = ((ln["bbox"][0] + ln["bbox"][2]) / 2, cy)
        elif cleaned in ("no", "n") and no_pos is None:
            no_pos = ((ln["bbox"][0] + ln["bbox"][2]) / 2, cy)

    if not yes_pos or not no_pos:
        return None

    # Check checkbox positions just LEFT of each Yes/No text
    yes_marked, yes_d = is_position_marked(img, yes_pos[0] - 12, yes_pos[1], size_pt=10.0, threshold=0.0)
    no_marked, no_d = is_position_marked(img, no_pos[0] - 12, no_pos[1], size_pt=10.0, threshold=0.0)

    # Also check positions slightly above (in case anchor is below text)
    if yes_d < 0.20:
        _, yd2 = is_position_marked(img, yes_pos[0] - 12, yes_pos[1] - 12, size_pt=10.0, threshold=0.0)
        yes_d = max(yes_d, yd2)
    if no_d < 0.20:
        _, nd2 = is_position_marked(img, no_pos[0] - 12, no_pos[1] - 12, size_pt=10.0, threshold=0.0)
        no_d = max(no_d, nd2)

    if max(yes_d, no_d) < 0.20:
        return None
    if yes_d > no_d * 1.15:
        return True
    if no_d > yes_d * 1.15:
        return False
    return None


def structure_page(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured fields from a page's controls and text lines.

    Returns dict with keys like sex, Smoker, Diabetic, allergies, symptoms, claim_type.

    When text was extracted via OCR (scanned documents), spatial tolerances
    are widened to account for OCR positioning noise. For OCR mode, also uses
    position-based mark detection as a fallback for Yes/No fields.
    """
    controls = page["controls"]
    lines = _index_by_text(page.get("lines", []))
    size = tuple(page.get("size", [612.0, 792.0]))
    page_w = size[0]
    page_mid = page_w / 2.0
    scale = 300 / 72.0

    # OCR-aware tolerance multiplier: widen search bands for scanned docs
    is_ocr = page.get("text_source") == "ocr"
    tol = 1.8 if is_ocr else 1.0

    by_label: dict[str, list[dict[str, Any]]] = {}
    for c in controls:
        lab = (c.get("label") or "").strip()
        by_label.setdefault(lab, []).append(c)

    result: dict[str, Any] = {}

    # --- Build row clusters (2-column layout) ---
    line_entries = []
    for text, (x0, y0, x1, y1) in lines:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        col = 0 if cx < page_mid else 1
        line_entries.append((text, (x0, y0, x1, y1), col, cy))

    rows: dict[int, list] = {0: [], 1: []}
    for col in (0, 1):
        col_lines = [(t, b, cy) for (t, b, ccol, cy) in line_entries if ccol == col]
        col_lines.sort(key=lambda t: t[2])
        current: list = []
        current_y = None
        for t, b, cy in col_lines:
            if current_y is None or abs(cy - current_y) <= 9:
                current.append((t, b, cy))
                current_y = cy if current_y is None else (current_y * (len(current) - 1) + cy) / len(current)
            else:
                rows[col].append((current_y, current))
                current = [(t, b, cy)]
                current_y = cy
        if current:
            rows[col].append((current_y, current))

    def infer_label_right_row(cy, cx, col, max_dx=140):
        if not rows[col]:
            return ""
        ry, rlines = min(rows[col], key=lambda rc: abs(rc[0] - cy))
        if abs(ry - cy) > 14:
            return ""
        best_dx = 1e9
        best = ""
        for t, (bx, by, ex, ey), _ in rlines:
            dx = bx - cx
            if 0 <= dx <= max_dx and dx < best_dx:
                best_dx = dx
                best = t.strip()
        return best

    def controls_in_row(ax0, ax1, band_y, max_dx=320, max_dy=28):
        """Find controls on the same row as an anchor, right of the anchor text.

        Restricts to controls in the same column as the anchor to avoid
        cross-column contamination (e.g. symptom checkboxes in col 1 being
        matched as Yes/No controls for a col 0 anchor).
        """
        anchor_col = 0 if (ax0 + ax1) / 2 < page_mid else 1
        effective_dy = max_dy * tol  # wider for OCR
        effective_dx = max_dx * tol
        hits = []
        for c in controls:
            if c["kind"] not in ("checkbox", "radio"):
                continue
            x, y, w, h = c["bbox"]
            cy = (y + h / 2) / scale
            cx = (x + w / 2) / scale
            cand_col = 0 if cx < page_mid else 1
            if cand_col != anchor_col:
                continue
            if abs(cy - band_y) <= effective_dy and -12 * tol <= (cx - ax1) <= effective_dx:
                hits.append((cx, cy, c))
        hits.sort(key=lambda t: t[0])
        return [c for _, _, c in hits]

    # --- Sex / Gender ---
    sex_anchor = _find_anchor_by_variants(lines, SEX_ANCHORS)
    if sex_anchor:
        _, (ax0, ay0, ax1, ay1) = sex_anchor
        band_y = (ay0 + ay1) / 2
        anchor_col = 0 if ax0 < page_mid else 1
        candidates = controls_in_row(ax0, ax1, band_y, max_dx=320, max_dy=26)

        # Find the selected radio/checkbox with a matching label
        for c in candidates:
            if c["selected"]:
                x, y, w, h = c["bbox"]
                cy = (y + h / 2) / scale
                cx = (x + w / 2) / scale
                lab = c.get("label") or infer_label_right_row(cy, cx, anchor_col, max_dx=60)
                if lab in ("Male", "Female", "Other"):
                    result["sex"] = lab
                    break

        # Fallback: ordered by x-position (Male, Female, Other assumed order)
        if "sex" not in result:
            selected = [c for c in candidates if c["selected"]]
            if selected:
                options = ["Male", "Female", "Other"]
                sorted_all = sorted(candidates, key=lambda c: c["bbox"][0])
                for c in selected:
                    idx = sorted_all.index(c)
                    if idx < len(options):
                        result["sex"] = options[idx]
                        break

        # Fallback 2: by_label direct match near anchor row
        if "sex" not in result:
            for option in ("Male", "Female", "Other"):
                for c in by_label.get(option, []):
                    if c.get("selected"):
                        x, y, w, h = c["bbox"]
                        cy = (y + h / 2) / scale
                        if abs(cy - band_y) <= 30:
                            result["sex"] = option
                            break
                if "sex" in result:
                    break

        # Fallback 3: on scans, check control labels after OCR cleanup
        if "sex" not in result and is_ocr:
            for c in candidates:
                if c["selected"]:
                    lab = _clean_ocr_text(c.get("label") or "").strip()
                    if lab.lower() in ("male", "m"):
                        result["sex"] = "Male"
                        break
                    elif lab.lower() in ("female", "f"):
                        result["sex"] = "Female"
                        break

        # Fallback 4: infer "Other" when anchor found but no selected sex control.
        if "sex" not in result and candidates:
            has_any_selected = any(c["selected"] for c in candidates if (c.get("label") or "") in ("Male", "Female", "Other"))
            if not has_any_selected:
                result["sex"] = "Other"

    # --- Yes/No fields ---
    # Labels that indicate a control belongs to a different field and should
    # NOT be matched as a yes/no control
    NON_YESNO_LABELS = (
        set(ALLERGY_OPTIONS)
        | set(SYMPTOMS)
        | set(CLAIM_OPTS)
        | {"Male", "Female", "Other"}
    )

    def resolve_yes_no(
        variants: tuple[str, ...],
        key_name: str,
        max_dx: float = 320,
        max_dy: float = 26,
    ):
        anchor = _find_anchor_by_variants(lines, variants)
        if not anchor:
            return
        _, (ax0, ay0, ax1, ay1) = anchor
        band_y = (ay0 + ay1) / 2

        band_controls = controls_in_row(ax0, ax1, band_y, max_dx=max_dx, max_dy=max_dy)
        # Exclude controls whose labels clearly identify them as another field
        band_controls = [c for c in band_controls if (c.get("label") or "") not in NON_YESNO_LABELS]
        if not band_controls:
            return

        # Cluster controls by y-coordinate (row), then pick the row closest to anchor.
        # This prevents neighboring yes/no rows (e.g. Smoker row above Diabetic) from
        # being merged when their vertical spacing is <= max_dy.
        rows_by_y: list[tuple[float, list]] = []
        for c in band_controls:
            cy = (c["bbox"][1] + c["bbox"][3] / 2) / scale
            placed = False
            for i, (ry, rc_list) in enumerate(rows_by_y):
                if abs(ry - cy) <= 6:
                    rc_list.append(c)
                    new_ry = (ry * len(rc_list) + cy) / (len(rc_list) + 1)
                    rows_by_y[i] = (new_ry, rc_list)
                    placed = True
                    break
            if not placed:
                rows_by_y.append((cy, [c]))

        # Pick the row closest to the anchor y
        rows_by_y.sort(key=lambda r: abs(r[0] - band_y))
        closest_row = rows_by_y[0][1]

        yes_labeled = [c for c in closest_row if (c.get("label") or "") == "Yes"]
        no_labeled = [c for c in closest_row if (c.get("label") or "") == "No"]
        yes_sel = any(c["selected"] for c in yes_labeled)
        no_sel = any(c["selected"] for c in no_labeled)

        # Fallback: leftmost two controls on the row are Yes/No in order
        if not yes_labeled or not no_labeled:
            if len(closest_row) >= 2:
                sorted_ctrls = sorted(closest_row, key=lambda c: c["bbox"][0])[:2]
                yes_sel = sorted_ctrls[0]["selected"]
                no_sel = sorted_ctrls[1]["selected"]

        if yes_sel or no_sel:
            if yes_sel and not no_sel:
                result[key_name] = True
            elif no_sel and not yes_sel:
                result[key_name] = False
            else:
                result[key_name] = None

    resolve_yes_no(("smoker",), "Smoker")
    resolve_yes_no(("diabetic",), "Diabetic")
    resolve_yes_no(WORK_RELATED_ANCHORS, "Is this work-related?")
    resolve_yes_no(AUTO_ACCIDENT_ANCHORS, "Auto accident?", max_dx=220, max_dy=20)

    # OCR: confidence-gated position override for Yes/No fields.
    # CV on scans produces values but often with borderline confidence (~0.5-0.6).
    # When CV confidence is low, position-based detection is more reliable.
    # When CV confidence is high (>0.7), trust CV.
    if is_ocr and pdf_path:
        for anchor, key in [
            ("smoker", "Smoker"),
            ("diabetic", "Diabetic"),
            ("work-related", "Is this work-related?"),
            ("auto accident", "Auto accident?"),
        ]:
            # Check if CV result exists AND has high confidence
            cv_confident = False
            if key in result and result[key] is not None:
                # Find the control that determined this value and check its conf
                for c in controls:
                    lab = (c.get("label") or "").strip()
                    if lab in ("Yes", "No") and c.get("selected") and c.get("conf", 0) >= 0.7:
                        cv_confident = True
                        break

            if not cv_confident:
                v = _ocr_position_yes_no(page, pdf_path, anchor)
                if v is not None:
                    result[key] = v

    # --- Allergies ---
    # Use text-first approach: find each allergy option text line below the anchor,
    # then look for a checkbox immediately to its left. This mirrors the symptoms
    # approach and is more robust than scanning a loose region.
    allergies: list[str] = []
    all_anchor = _find_anchor_by_variants(lines, ALLERGIES_ANCHORS)
    if all_anchor:
        _, (ax0, ay0, ax1, ay1) = all_anchor
        band_y = (ay0 + ay1) / 2

        # Collect allergy-related text lines below the anchor (any column, since
        # allergies might span sub-columns in grid layouts).
        option_hits: list[tuple[str, str, tuple[float, float, float, float]]] = []
        # Lower threshold for OCR mode — labels may be partially garbled
        allergy_threshold = 0.55 if is_ocr else 0.8
        for text, (x0, y0, x1, y1) in lines:
            cy = (y0 + y1) / 2
            if cy - band_y < 4 or cy - band_y > 220:
                continue
            for a in ALLERGY_OPTIONS:
                if _token_set_ratio(text, a) >= allergy_threshold:
                    option_hits.append((a, text, (x0, y0, x1, y1)))
                    break

        for allergy_name, _line_text, (bx0, by0, bx1, by1) in option_hits:
            ly = (by0 + by1) / 2
            # Find nearest checkbox to the LEFT of the option text, same row
            found = None
            best_dx = 1e9
            all_dy = 12 * tol
            all_dx = 35 * tol
            for c in controls:
                x, y, w, h = c["bbox"]
                cy = (y + h / 2) / scale
                cx = (x + w / 2) / scale
                dx = bx0 - cx
                if abs(cy - ly) <= all_dy and 0 <= dx <= all_dx and dx < best_dx:
                    found = c
                    best_dx = dx
            if (
                found
                and found.get("selected")
                and found.get("conf", 1.0) >= 0.51
                and allergy_name not in allergies
            ):
                allergies.append(allergy_name)

    if allergies:
        result["allergies"] = allergies
    else:
        # Fallback: direct label match
        fallback = []
        for a in ALLERGY_OPTIONS:
            for c in by_label.get(a, []):
                if c.get("selected") and c.get("conf", 1.0) >= 0.51:
                    fallback.append(a)
                    break
        if fallback:
            result["allergies"] = fallback

    # --- Symptoms ---
    symptoms: list[str] = []
    sym_anchor = _find_anchor_by_variants(lines, SYMPTOMS_ANCHORS)
    if sym_anchor:
        _, (ax0, ay0, ax1, ay1) = sym_anchor
        band_y = (ay0 + ay1) / 2
        anchor_col = 0 if ax0 < page_mid else 1

        # Collect lines below the anchor in same column
        col_lines = [
            (t, b) for (t, b, col, cy) in line_entries
            if col == anchor_col and ((b[1] + b[3]) / 2) > band_y
        ]

        # Lower threshold for OCR mode
        sym_threshold = 0.5 if is_ocr else 0.7
        for s in SYMPTOMS:
            candidates = [(t, b) for t, b in col_lines if _token_set_ratio(t, s) >= sym_threshold]
            if not candidates:
                continue

            t, (bx0, by0, bx1, by1) = sorted(candidates, key=lambda tb: tb[1][1])[0]
            ly = (by0 + by1) / 2

            # Find nearest checkbox to the LEFT of the symptom label text, same column.
            # Tight max_dx: checkboxes are typically ~15px left of their text label.
            # Wider tolerances for OCR to handle position noise.
            found = None
            best_dx = 1e9
            sym_dy = 12 * tol
            sym_dx = 35 * tol
            for c in controls:
                x, y, w, h = c["bbox"]
                cy = (y + h / 2) / scale
                cx = (x + w / 2) / scale
                cand_col = 0 if cx < page_mid else 1
                if cand_col != anchor_col:
                    continue
                dx = bx0 - cx
                if abs(cy - ly) <= sym_dy and 0 <= dx <= sym_dx and dx < best_dx:
                    found = c
                    best_dx = dx

            # Grid fields need stronger conf to avoid false positives near anchors
            if (
                found
                and found.get("selected")
                and found.get("conf", 1.0) >= 0.51
                and s not in symptoms
            ):
                symptoms.append(s)

    if symptoms:
        result["symptoms"] = symptoms
    else:
        # Fallback: fuzzy match against all labels
        fallback = []
        for s in SYMPTOMS:
            best = 0.0
            for lab, cs in by_label.items():
                if not lab:
                    continue
                sc = _token_set_ratio(lab, s)
                if sc > best and any(c.get("selected") for c in cs):
                    best = sc
            if best >= 0.5:
                fallback.append(s)
        if fallback:
            result["symptoms"] = fallback

    # On scans: OCR text pattern detection for allergy/symptom grids.
    # OCR reads checkbox marks as text patterns: "(X) Pollen", "|| Penicillin",
    # "J Peanuts" for selected, "() Shellfish", "Latex" for unselected.
    if is_ocr and pdf_path:
        import re as _ocr_re
        from extracto.structuring.fuzzy import fuzzy_contains as _fc

        _sel_pat = _ocr_re.compile(
            r"^[\[\(C]*[XxKk✓☑][\]\)]*\s*|"   # (X), [X], CX)
            r"^\|[\|I!l]\|?\s*|"                # || or |I| (filled box)
            r"^[J]\s+|"                          # J (filled square glyph)
            r"^\[_\]\s*|"                        # [_] dark bracket
            r"^[☐☑✗✓]\s*"                       # unicode checkmarks
        )

        def _ocr_detect(opts: list[str], anchors: list[str]) -> list[str]:
            anchor_y = None
            for anc in anchors:
                for ln_d in page.get("lines", []):
                    t = ln_d.get("text", "")
                    if anc.lower() in t.lower() or _fc(t, anc, threshold=0.7):
                        anchor_y = ln_d["bbox"][1]
                        break
                if anchor_y is not None:
                    break
            if anchor_y is None:
                return []
            found = []
            for opt in opts:
                for ln_d in page.get("lines", []):
                    cy = (ln_d["bbox"][1] + ln_d["bbox"][3]) / 2
                    if cy < anchor_y:
                        continue
                    text = ln_d.get("text", "").strip()
                    if opt.lower() not in text.lower() and not _fc(text, opt, threshold=0.75):
                        continue
                    if _sel_pat.match(text):
                        found.append(opt)
                    break
            return found

        # Only use OCR pattern if CV didn't find anything useful
        if not result.get("allergies"):
            ocr_all = _ocr_detect(ALLERGY_OPTIONS, ["allergies", "allergy", "check all"])
            if ocr_all:
                result["allergies"] = sorted(set(ocr_all))

        if not result.get("symptoms"):
            ocr_sym = _ocr_detect(SYMPTOMS, ["current symptoms", "symptoms", "symptom"])
            if ocr_sym:
                result["symptoms"] = sorted(set(ocr_sym))

    # --- Claim type ---
    claim_anchor = _find_line(lines, lambda t: t.strip().lower().startswith("claim type"))
    found_claim = False
    if claim_anchor:
        _, (ax0, ay0, ax1, ay1) = claim_anchor
        band_y = (ay0 + ay1) / 2
        anchor_col = 0 if ax0 < page_mid else 1
        for o in CLAIM_OPTS:
            for c in by_label.get(o, []):
                x, y, w, h = c["bbox"]
                cy = (y + h / 2) / scale
                cx = (x + w / 2) / scale
                cand_col = 0 if cx < page_mid else 1
                if cand_col == anchor_col and abs(cy - band_y) < 120 and -20 < (cx - ax1) < 260 and c["selected"]:
                    result["claim_type"] = o
                    found_claim = True
                    break
            if found_claim:
                break
    if not found_claim:
        for o in CLAIM_OPTS:
            if any(c["selected"] for c in by_label.get(o, [])):
                result["claim_type"] = o
                break

    return result
