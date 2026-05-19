"""Parse Explanation of Benefits (EOB) forms.

Distinctive features this module handles:
1. Wide financial table with 8+ columns
2. Column detection via header-row x-positions (not hardcoded)
3. Row-based cell alignment using y-band clustering
4. TOTALS row parsing (excluded from service lines)
5. Reason code extraction from comma-separated cells
6. Reason code legend lookup at the bottom
"""

from __future__ import annotations

import re
from typing import Any

# Column identifiers we care about in the service line table
TABLE_COLUMNS = [
    ("date_of_service", "Date"),
    ("cpt", "CPT"),
    ("billed", "Billed"),
    ("allowed", "Allowed"),
    ("deductible", "Deductible"),
    ("copay", "Copay"),
    ("coinsurance", "Coins"),
    ("plan_paid", "Plan Paid"),
    ("patient_resp", "Pt Resp"),
    ("reason", "Rsn"),
]

DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
CURRENCY_RE = re.compile(r"\$?(\d+(?:,\d{3})*\.\d{2})")
CPT_RE = re.compile(r"^\d{5}$")
REASON_CODE_RE = re.compile(r"\b([A-Z]{2}-\d{1,4})\b")


def _parse_currency(text: str) -> float | None:
    m = CURRENCY_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


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
            if ":" in ln["text"] and ":" not in label_text:
                continue
            best = ln["text"].strip()
            best_dx = dx
    return best


def _row_at_y(lines, y: float, tolerance: float = 3.0) -> list[dict]:
    """Return all text lines whose center y is within tolerance of `y`."""
    row = []
    for ln in lines:
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        if abs(cy - y) <= tolerance:
            row.append(ln)
    row.sort(key=lambda ln: ln["bbox"][0])
    return row


def _cluster_rows(lines: list[dict], y_tol: float = 3.0) -> list[tuple[float, list[dict]]]:
    """Group text lines by y-coordinate into rows."""
    sorted_lines = sorted(lines, key=lambda ln: (ln["bbox"][1] + ln["bbox"][3]) / 2)
    rows: list[tuple[float, list[dict]]] = []
    for ln in sorted_lines:
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        placed = False
        for i, (ry, row_list) in enumerate(rows):
            if abs(ry - cy) <= y_tol:
                row_list.append(ln)
                new_ry = (ry * len(row_list) + cy) / (len(row_list) + 1)
                rows[i] = (new_ry, row_list)
                placed = True
                break
        if not placed:
            rows.append((cy, [ln]))
    for i, (ry, row_list) in enumerate(rows):
        row_list.sort(key=lambda ln: ln["bbox"][0])
        rows[i] = (ry, row_list)
    return rows


def extract_header_fields(lines) -> dict[str, Any]:
    """Extract the header key-value fields (claim number, check number, member name, etc.)."""
    result: dict[str, Any] = {}

    claim = _nearest_text_right(lines, "claim number", max_dx=250)
    if claim:
        result["claim_number"] = claim

    check_num = _nearest_text_right(lines, "check number", max_dx=250)
    if check_num:
        result["check_number"] = check_num

    member = _nearest_text_right(lines, "member name", max_dx=300)
    if member:
        result["member_name"] = member

    return result


def extract_payer_name(lines) -> str | None:
    """The payer name is typically the first prominent line below the EOB title."""
    title_ln = _find_line_contains(lines, "explanation of benefits")
    if not title_ln:
        return None
    title_y = title_ln["bbox"][3]

    # Find text lines in the 40pt below the title, in the left column (x < 300)
    candidates = []
    for ln in lines:
        if "not a bill" in ln["text"].lower():
            continue
        if "explanation of benefits" in ln["text"].lower():
            continue
        bbox = ln["bbox"]
        cy = (bbox[1] + bbox[3]) / 2
        if title_y < cy <= title_y + 30 and bbox[0] < 300:
            candidates.append((cy, bbox[0], ln["text"].strip()))

    if not candidates:
        return None

    # Pick the topmost one
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


def extract_service_lines_and_totals(lines) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Parse the CLAIM DETAIL table.

    Returns:
        (service_lines, totals) where:
        - service_lines is a list of dicts with date_of_service, cpt, billed,
          allowed, deductible, copay, coinsurance, plan_paid, patient_resp,
          reason_codes
        - totals is a dict with the sum of each currency column
    """
    # Find the header row containing column labels
    header_ln = None
    for ln in lines:
        text_lower = ln["text"].lower()
        if "billed" in text_lower and "allowed" in text_lower:
            # This might be a merged multi-header line
            header_ln = ln
            break

    # Alternate: look for the individual column labels on a single y row
    if not header_ln:
        billed_ln = _find_line_contains(lines, "billed")
        if billed_ln:
            header_ln = billed_ln

    if not header_ln:
        return [], {}

    header_y = (header_ln["bbox"][1] + header_ln["bbox"][3]) / 2

    # Collect column header x-positions from the same y band
    header_row = _row_at_y(lines, header_y, tolerance=3.0)
    col_positions: dict[str, float] = {}
    for ln in header_row:
        t = ln["text"].strip().lower()
        x = ln["bbox"][0]
        # Map header text to canonical column name
        if "date" in t and "service" not in t[:5]:
            col_positions["date_of_service"] = x
        elif t.startswith("date"):
            col_positions["date_of_service"] = x
        if t == "cpt" or "cpt" in t[-4:]:
            # Avoid picking "Date of ServiceCPT" at the date column x
            if "date_of_service" in col_positions and abs(x - col_positions["date_of_service"]) < 5:
                # Merged cell - assume CPT is at a fixed offset
                col_positions["cpt"] = x + 55
            else:
                col_positions["cpt"] = x
        if t == "billed" or t.startswith("billed"):
            col_positions["billed"] = x
        if t == "allowed" or t.startswith("allowed"):
            col_positions["allowed"] = x
        if "deductible" in t:
            col_positions["deductible"] = x
        if t == "copay":
            col_positions["copay"] = x
        if "coins" in t:
            col_positions["coinsurance"] = x
        if "plan paid" in t or "plan" in t:
            col_positions["plan_paid"] = x
        if "pt resp" in t or "patient" in t:
            col_positions["patient_resp"] = x
        if t == "rsn" or "reason" in t:
            col_positions["reason"] = x

    if not col_positions:
        return [], {}

    # Build ordered list of (column_name, x_position) for column assignment
    ordered_cols = sorted(col_positions.items(), key=lambda c: c[1])

    def assign_column(x: float) -> str | None:
        """Given an x-coordinate, return the closest column name."""
        best = None
        best_dist = 1e9
        for col_name, col_x in ordered_cols:
            d = abs(x - col_x)
            if d < best_dist and d <= 40:
                best = col_name
                best_dist = d
        return best

    # Now find all rows below the header that look like data rows
    row_clusters = _cluster_rows(lines, y_tol=3.0)

    # Filter to rows below the header
    data_rows = [(ry, rl) for ry, rl in row_clusters if ry > header_y + 8]

    service_lines: list[dict[str, Any]] = []
    totals: dict[str, float] = {}

    for ry, row_list in data_rows:
        row_dict: dict[str, Any] = {}
        is_totals = False

        for ln in row_list:
            text = ln["text"].strip()
            if text.upper() == "TOTALS":
                is_totals = True
                continue
            x = ln["bbox"][0]
            col = assign_column(x)
            if not col:
                continue

            # Parse value based on column type
            if col == "date_of_service":
                m = DATE_RE.search(text)
                if m:
                    row_dict[col] = m.group(1)
            elif col == "cpt":
                if CPT_RE.match(text):
                    row_dict[col] = text
            elif col == "reason":
                codes = REASON_CODE_RE.findall(text)
                if codes:
                    row_dict[col] = codes
            else:
                val = _parse_currency(text)
                if val is not None:
                    row_dict[col] = val

        if is_totals:
            # Store totals
            for key in ("billed", "allowed", "deductible", "copay", "coinsurance", "plan_paid", "patient_resp"):
                if key in row_dict:
                    totals[key] = row_dict[key]
            # Stop processing data rows after totals
            break
        else:
            # Only include if this looks like a service line (has a CPT or DOS)
            if "cpt" in row_dict or "date_of_service" in row_dict:
                service_lines.append(row_dict)

    return service_lines, totals


def extract_reason_codes_used(service_lines: list[dict]) -> list[str]:
    """Collect all unique reason codes from parsed service lines."""
    codes: set[str] = set()
    for line in service_lines:
        for code in line.get("reason", []):
            codes.add(code)
    return sorted(codes)


def extract_patient_responsibility_summary(lines) -> dict[str, float]:
    """Extract totals from the Patient Responsibility Summary box.

    On scanned EOBs, the table is often garbled by OCR but the summary box
    at the bottom is clean because it's rendered in larger, isolated text:

        Deductible:         $103.15
        Copay:              $30.00
        Coinsurance:        $0.00
        Total Patient Owes: $133.15

    This provides a reliable fallback for total_patient_resp and its
    component parts when table parsing fails.
    """
    summary: dict[str, float] = {}
    in_summary = False

    for ln in lines:
        text = ln.get("text", "")
        text_lower = text.lower()
        if "patient responsibility" in text_lower or "patient owes" in text_lower:
            in_summary = True
        if not in_summary:
            continue

        # Look for "Label: $amount" patterns in same or nearby lines
        for label_key, field_key in [
            ("deductible", "deductible"),
            ("copay", "copay"),
            ("coinsurance", "coinsurance"),
            ("total patient owes", "patient_resp"),
            ("total patient", "patient_resp"),
            ("plan paid", "plan_paid"),
        ]:
            if label_key in text_lower:
                m = CURRENCY_RE.search(text)
                if m:
                    try:
                        summary[field_key] = float(m.group(1).replace(",", ""))
                    except ValueError:
                        pass
                    break
                # Currency might be on the next line or a nearby line at same y
                bbox = ln.get("bbox")
                if bbox:
                    for ln2 in lines:
                        if ln2 is ln:
                            continue
                        b2 = ln2.get("bbox")
                        if b2 and abs((b2[1] + b2[3]) / 2 - (bbox[1] + bbox[3]) / 2) <= 4:
                            m2 = CURRENCY_RE.search(ln2.get("text", ""))
                            if m2:
                                try:
                                    summary[field_key] = float(m2.group(1).replace(",", ""))
                                except ValueError:
                                    pass
                                break

    return summary


def extract_totals_from_currency_lines(lines) -> dict[str, float]:
    """Fallback: extract total_billed, total_plan_paid from TOTALS row.

    Even when column-based parsing fails, the TOTALS row may have
    currency values we can match by position or pattern.
    """
    totals: dict[str, float] = {}
    for ln in lines:
        text = ln.get("text", "")
        if "totals" in text.lower() or "total" in text.lower():
            # Find all currency values on or near this line
            currencies = CURRENCY_RE.findall(text)
            for c in currencies:
                try:
                    val = float(c.replace(",", ""))
                    # The largest currency on a TOTALS line is likely total_billed
                    if val > totals.get("billed", 0):
                        totals["billed"] = val
                except ValueError:
                    pass
    return totals


def _extract_totals_from_text(lines) -> dict[str, float]:
    """Find TOTALS row values by scanning for the word TOTALS near currency values."""
    for ln in lines:
        text = ln.get("text", "")
        if "total" in text.lower() and "$" in text:
            # This line contains TOTALS + currency values
            amounts = CURRENCY_RE.findall(text)
            values = []
            for a in amounts:
                try:
                    values.append(float(a.replace(",", "")))
                except ValueError:
                    pass
            if values:
                # The standard EOB order: Billed, Allowed, Deductible, Copay, Coins, PlanPaid, PtResp
                result = {}
                if len(values) >= 1:
                    result["billed"] = max(values)  # largest is typically total_billed
                if len(values) >= 2:
                    sorted_vals = sorted(values, reverse=True)
                    result["allowed"] = sorted_vals[1] if len(sorted_vals) > 1 else 0
                return result
    return {}


def structure_eob(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured fields from an EOB form."""
    lines = page.get("lines", [])
    is_ocr = page.get("text_source") == "ocr"

    result: dict[str, Any] = {"form_type": "eob"}

    # Header fields
    payer = extract_payer_name(lines)
    if payer:
        result["payer_name"] = payer

    header_fields = extract_header_fields(lines)
    result.update(header_fields)

    # Service line table + totals
    service_lines, totals = extract_service_lines_and_totals(lines)

    if service_lines:
        result["service_lines"] = service_lines
        result["service_line_count"] = len(service_lines)
        result["service_line_cpts"] = [l["cpt"] for l in service_lines if "cpt" in l]

    if totals:
        if "billed" in totals:
            result["total_billed"] = totals["billed"]
        if "allowed" in totals:
            result["total_allowed"] = totals["allowed"]
        if "plan_paid" in totals:
            result["total_plan_paid"] = totals["plan_paid"]
        if "patient_resp" in totals:
            result["total_patient_resp"] = totals["patient_resp"]

    # FALLBACK for scanned EOBs: Patient Responsibility Summary box
    if is_ocr or "total_patient_resp" not in result:
        summary = extract_patient_responsibility_summary(lines)
        if summary:
            if "patient_resp" in summary and "total_patient_resp" not in result:
                result["total_patient_resp"] = summary["patient_resp"]
            if "plan_paid" in summary and "total_plan_paid" not in result:
                result["total_plan_paid"] = summary["plan_paid"]

    # FALLBACK 2: Enhanced OCR at 600 DPI with CLAHE contrast enhancement.
    # The standard 300 DPI OCR can't read the dense 8pt table text, but
    # 600 DPI + histogram equalization recovers some dollar values.
    if is_ocr and pdf_path and ("total_billed" not in result or "service_line_cpts" not in result):
        try:
            import fitz as _fitz2
            import cv2 as _cv2
            import numpy as _np2

            _doc2 = _fitz2.open(pdf_path)
            _page2 = _doc2[0]
            _dpi2 = 600
            _mat2 = _fitz2.Matrix(_dpi2 / 72, _dpi2 / 72)
            _pix2 = _page2.get_pixmap(matrix=_mat2, alpha=False)
            _img2 = _np2.frombuffer(_pix2.samples, dtype=_np2.uint8).reshape(_pix2.height, _pix2.width, 3)

            _gray2 = _cv2.cvtColor(_img2, _cv2.COLOR_BGR2GRAY)
            _clahe = _cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            _enhanced = _clahe.apply(_gray2)
            _, _binary = _cv2.threshold(_enhanced, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
            _bgr = _cv2.cvtColor(_binary, _cv2.COLOR_GRAY2BGR)

            _ok, _buf = _cv2.imencode(".png", _bgr)
            if _ok:
                _new_doc = _fitz2.open()
                _h, _w = _bgr.shape[:2]
                _pw, _ph = _w * 72 / _dpi2, _h * 72 / _dpi2
                _new_page = _new_doc.new_page(width=_pw, height=_ph)
                _new_page.insert_image(_fitz2.Rect(0, 0, _pw, _ph), stream=_buf.tobytes())
                _tp2 = _new_page.get_textpage_ocr(language="eng", dpi=_dpi2, full=True)
                _text2 = _new_page.get_text(textpage=_tp2)

                # Extract garbled dollar values (OCR reads $ as s on some forms)
                import re as _re2
                _garbled = _re2.findall(r"[sS](\d+\.\d{2})", _text2)
                _real = _re2.findall(r"\$(\d[\d,.]+)", _text2)
                _all_amounts = []
                for _a in _garbled + _real:
                    try:
                        _all_amounts.append(float(_a.replace(",", "")))
                    except ValueError:
                        pass

                # Find CPTs from enhanced OCR
                _cpts = list(set(_re2.findall(r"\b(\d{5})\b", _text2)))
                _valid_cpts = [c for c in _cpts if 10000 <= int(c) <= 99999]

                if _valid_cpts and "service_line_cpts" not in result:
                    result["service_line_cpts"] = sorted(_valid_cpts)
                    result["service_line_count"] = len(_valid_cpts)

                pass  # Enhanced OCR dollar values are unreliable for table fields

                _new_doc.close()
            _doc2.close()
        except Exception:
            pass

    # FALLBACK 3: TableTransformer — detect table structure, OCR each cell individually.
    # This breaks through the dense-table OCR ceiling by isolating cells before OCR.
    if is_ocr and pdf_path and (
        "service_line_cpts" not in result or "total_billed" not in result
    ):
        try:
            from extracto.detection.table_extract import extract_table_cells_subprocess, parse_eob_table
            tables = extract_table_cells_subprocess(pdf_path)
            # Only use CPTs from table (dollar values are too garbled to trust)
            if tables:
                table_data = parse_eob_table(tables)
                if "service_line_cpts" in table_data:
                    pass  # Will be merged below
            if tables:
                table_data = parse_eob_table(tables)
                # Merge table-extracted fields that we don't have yet
                for key in ("service_line_cpts", "service_line_count", "total_billed",
                            "total_allowed", "total_plan_paid"):
                    if key in table_data and key not in result:
                        result[key] = table_data[key]
                # Supplement reason codes
                extra_codes = table_data.get("reason_codes_from_table", [])
                if extra_codes:
                    existing = set(result.get("reason_codes_used", []))
                    result["reason_codes_used"] = sorted(existing | set(extra_codes))
        except Exception:
            pass

    # Currency extraction from full text is unreliable on scans — disabled.
    # The patient_resp from the Summary box is the only reliable currency source.

    # Reason codes used across all lines
    result["reason_codes_used"] = extract_reason_codes_used(service_lines)

    # Reason codes fallback: scan full text for CARC codes
    if not result["reason_codes_used"]:
        full_text = " ".join(ln.get("text", "") for ln in lines)
        codes = REASON_CODE_RE.findall(full_text)
        if codes:
            result["reason_codes_used"] = sorted(set(codes))

    return result
