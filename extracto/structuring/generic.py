"""Generic document extractor — works on ANY document without form-specific logic.

When a document doesn't match any known form type, this module extracts
whatever structured data it can find using universal heuristics:

1. **Key-value pairs**: "Label: Value" patterns, "Label _____ Value" patterns
2. **Checkbox/radio selections**: All detected controls with their labels
3. **Tables**: Rows of column-aligned text
4. **Dates with context**: Every date found with its surrounding label text
5. **Entities**: Patient names, phone numbers, addresses, SSN patterns, MRNs
6. **Section headers**: Bold/large text that appears to be a section boundary

This ensures every document returns useful structured output — a law firm
never gets an empty result.
"""

from __future__ import annotations

import re
from typing import Any

import fitz

from extracto.detection.marks import find_marks, find_overlaid_text


# --- Patterns ---

# Key-value: "Label: Value" or "Label Value" where label is title-case
KV_RE = re.compile(
    r"^([A-Z][A-Za-z\s/&'.()-]{2,40}):\s+(.{1,200})$"
)

# Dates
DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

# Phone numbers
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

# SSN (last 4 or full with mask)
SSN_RE = re.compile(r"(?:XXX-XX-|xxx-xx-|\*{3}-\*{2}-)?\d{4}")
SSN_FULL_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# MRN patterns
MRN_RE = re.compile(r"\b(?:MRN|MR#|Medical Record)\s*[:#]?\s*([A-Z0-9-]{4,15})\b", re.I)

# Email
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Currency
CURRENCY_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")

# ICD-10
ICD10_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.[A-Z0-9]{1,5})?)\b")

# CPT
CPT_RE = re.compile(r"\b(\d{5})\b")

# NPI (10 digits)
NPI_RE = re.compile(r"\bNPI[:\s#]*(\d{10})\b", re.I)


def extract_key_value_pairs(lines: list[dict]) -> list[dict[str, str]]:
    """Extract 'Label: Value' patterns from text lines.

    Handles both single-line ('Label: Value') and multi-line patterns
    where the label ends with ':' on one line and the value is on the next.
    """
    pairs = []
    line_texts = [ln.get("text", "").strip() for ln in lines]

    for i, text in enumerate(line_texts):
        # Single-line: "Label: Value"
        m = KV_RE.match(text)
        if m:
            label = m.group(1).strip()
            value = m.group(2).strip()
            if len(value) > 0 and not value.startswith("_"):
                pairs.append({"label": label, "value": value})
            continue

        # Multi-line: "Label:" on this line, value on the next
        if text.endswith(":") and i + 1 < len(line_texts):
            label = text[:-1].strip()
            value = line_texts[i + 1].strip()
            # Only accept if label looks like a field name and value isn't another label
            if (len(label) >= 2 and len(value) >= 1
                    and not value.endswith(":")
                    and not value.startswith("_")):
                pairs.append({"label": label, "value": value})

    return pairs


def extract_selected_controls(controls: list[dict]) -> list[dict]:
    """Return all controls that are selected, with their labels and positions."""
    selected = []
    for c in controls:
        if c.get("selected"):
            selected.append({
                "kind": c["kind"],
                "label": c.get("label", ""),
                "confidence": round(c.get("conf", 0), 3),
            })
    return selected


def extract_dates_with_context(lines: list[dict]) -> list[dict]:
    """Find every date on the page with its surrounding text context."""
    dates = []
    seen = set()
    for ln in lines:
        text = ln.get("text", "")
        for m in DATE_RE.finditer(text):
            date_str = m.group(1)
            if date_str in seen:
                continue
            seen.add(date_str)

            # Get context: the label text around the date
            start = max(0, m.start() - 40)
            context = text[start:m.start()].strip()
            # Clean up context
            context = re.sub(r"[_\s]+$", "", context).strip()
            if context.endswith(":"):
                context = context[:-1].strip()

            dates.append({
                "date": date_str,
                "context": context if context else None,
            })
    return dates


def extract_entities(full_text: str) -> dict[str, list[str]]:
    """Extract recognizable entities: phones, emails, MRNs, NPIs, currencies."""
    entities: dict[str, list[str]] = {}

    phones = list(set(PHONE_RE.findall(full_text)))
    if phones:
        entities["phone_numbers"] = phones[:5]

    emails = list(set(EMAIL_RE.findall(full_text)))
    if emails:
        entities["emails"] = emails[:5]

    mrns = list(set(MRN_RE.findall(full_text)))
    if mrns:
        entities["mrn"] = mrns[:3]

    npis = list(set(NPI_RE.findall(full_text)))
    if npis:
        entities["npi"] = npis[:3]

    ssns = list(set(SSN_FULL_RE.findall(full_text)))
    if ssns:
        entities["ssn_detected"] = [f"***-**-{s[-4:]}" for s in ssns[:3]]

    currencies = list(set(CURRENCY_RE.findall(full_text)))
    if currencies:
        # Sort by value descending
        try:
            currencies.sort(key=lambda c: float(c.replace("$", "").replace(",", "")), reverse=True)
        except ValueError:
            pass
        entities["currency_values"] = currencies[:10]

    icd_codes = list(set(ICD10_RE.findall(full_text)))
    if icd_codes:
        entities["icd10_codes"] = sorted(icd_codes)[:15]

    return entities


def extract_tables(lines: list[dict]) -> list[dict]:
    """Detect tabular data by finding rows with column-aligned text.

    Groups text lines by y-coordinate into rows, then identifies rows that
    share a consistent column structure (similar x-positions across rows).
    """
    if not lines:
        return []

    # Filter to lines that have bbox (unit tests may pass lines without)
    lines_with_bbox = [l for l in lines if "bbox" in l]
    if not lines_with_bbox:
        return []

    # Cluster lines by y into rows
    rows: list[tuple[float, list[dict]]] = []
    sorted_lines = sorted(lines_with_bbox, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)

    for ln in sorted_lines:
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        placed = False
        for i, (ry, items) in enumerate(rows):
            if abs(ry - cy) <= 3:
                items.append(ln)
                rows[i] = ((ry * len(items) + cy) / (len(items) + 1), items)
                placed = True
                break
        if not placed:
            rows.append((cy, [ln]))

    # Find sequences of rows with 3+ cells each (likely a table)
    table_rows = []
    current_table: list[tuple[float, list]] = []

    for ry, items in rows:
        if len(items) >= 3:
            current_table.append((ry, items))
        else:
            if len(current_table) >= 2:
                table_rows.append(current_table)
            current_table = []
    if len(current_table) >= 2:
        table_rows.append(current_table)

    # Convert to structured output
    tables = []
    for table in table_rows:
        parsed_rows = []
        for ry, items in table:
            items.sort(key=lambda l: l["bbox"][0])
            cells = [l["text"].strip() for l in items]
            parsed_rows.append(cells)

        if parsed_rows:
            tables.append({
                "row_count": len(parsed_rows),
                "col_count": max(len(r) for r in parsed_rows),
                "header": parsed_rows[0] if parsed_rows else [],
                "rows": parsed_rows[1:] if len(parsed_rows) > 1 else [],
            })

    return tables


def extract_section_headers(page: fitz.Page) -> list[str]:
    """Find section headers by detecting text rendered in a larger/bolder font."""
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return []

    # Collect all spans with their font size
    spans: list[tuple[float, float, str, str]] = []
    for block in text_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 10)
                font = span.get("font", "")
                y = span.get("bbox", (0, 0, 0, 0))[1]
                if text and len(text) > 2:
                    spans.append((size, y, text, font))

    if not spans:
        return []

    # Body text is the most common font size
    from collections import Counter
    size_counts = Counter(round(s[0], 1) for s in spans)
    body_size = size_counts.most_common(1)[0][0]

    # Headers are significantly larger than body OR bold
    headers = []
    for size, y, text, font in spans:
        is_larger = size > body_size + 1.5
        is_bold = "bold" in font.lower() or "Black" in font
        if (is_larger or is_bold) and len(text) > 3 and len(text) < 80:
            # Skip pure number/date lines
            if not re.match(r"^[\d/.$,\s]+$", text):
                headers.append(text)

    return headers[:20]


def extract_acroform_fields(pdf_path: str) -> list[dict[str, str]]:
    """Extract filled values from AcroForm (fillable PDF) widgets.

    Many real government forms (SSA-827, ACORD 25, etc.) are fillable PDFs
    where text lives in form widgets, not the text layer. page.get_text()
    returns nothing, but page.widgets() returns the field names and values.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    fields = []
    for page in doc:
        for widget in (page.widgets() or []):
            name = widget.field_name or ""
            value = widget.field_value or ""
            field_type = widget.field_type_string or ""
            if name and value:
                fields.append({
                    "field_name": name,
                    "field_value": value,
                    "field_type": field_type,
                })
            elif name and field_type in ("CheckBox", "RadioButton"):
                # Report checkbox/radio state even without text value
                fields.append({
                    "field_name": name,
                    "field_value": "checked" if widget.field_value else "unchecked",
                    "field_type": field_type,
                })
    doc.close()
    return fields


def structure_generic(page: dict[str, Any], pdf_path: str | None = None) -> dict[str, Any]:
    """Extract structured data from ANY document using universal heuristics.

    Returns a best-effort extraction with whatever the document contains.
    Every document returns at least some useful data.

    For fillable PDFs (AcroForm), also extracts form widget field names and values.
    """
    controls = page.get("controls", [])
    lines = page.get("lines", [])
    full_text = " ".join(ln.get("text", "") for ln in lines)

    result: dict[str, Any] = {
        "form_type": "generic",
        "extraction_mode": "generic",
    }

    # Key-value pairs (most useful for structured forms)
    kv_pairs = extract_key_value_pairs(lines)
    if kv_pairs:
        result["key_value_pairs"] = kv_pairs
        # Promote common fields to top-level for convenience
        kv_dict = {p["label"].lower(): p["value"] for p in kv_pairs}
        for name_key in ("patient name", "patient's name", "name", "member name", "employee name"):
            if name_key in kv_dict:
                result["patient_name"] = kv_dict[name_key]
                break
        for dob_key in ("date of birth", "dob", "birth date"):
            if dob_key in kv_dict:
                result["patient_dob"] = kv_dict[dob_key]
                break

    # Selected checkboxes/radios
    selected = extract_selected_controls(controls)
    if selected:
        result["selected_controls"] = selected

    # Dates with context
    dates = extract_dates_with_context(lines)
    if dates:
        result["dates"] = dates

    # Named entities
    entities = extract_entities(full_text)
    if entities:
        result["entities"] = entities

    # Tables
    tables = extract_tables(lines)
    if tables:
        result["tables"] = tables

    # Section headers (requires PDF path for font info)
    if pdf_path:
        try:
            doc = fitz.open(pdf_path)
            headers = extract_section_headers(doc[0])
            doc.close()
            if headers:
                result["section_headers"] = headers
        except Exception:
            pass

    # AcroForm fields (fillable PDFs where text is in widgets, not text layer)
    if pdf_path:
        acroform = extract_acroform_fields(pdf_path)
        if acroform:
            result["acroform_fields"] = acroform
            # Also extract key-value pairs from AcroForm field names/values
            for field in acroform:
                if field["field_value"] and field["field_value"] not in ("checked", "unchecked"):
                    kv_pairs.append({
                        "label": field["field_name"],
                        "value": field["field_value"],
                    })
            # Re-enrich full_text with AcroForm values for entity extraction
            acro_text = " ".join(f["field_value"] for f in acroform if f["field_value"])
            if acro_text:
                combined_text = full_text + " " + acro_text
                extra_entities = extract_entities(combined_text)
                for k, v in extra_entities.items():
                    if k not in entities:
                        entities[k] = v
                if extra_entities:
                    result["entities"] = entities

    # --- Clinical note enrichment ---
    # Pull patient name, DOB, provider from clinical note patterns.
    # Handles both same-line ("Patient: Name") and next-line ("Patient:\nName") formats.
    line_texts = [ln.get("text", "").strip() for ln in lines]

    if "patient_name" not in result:
        # Try multi-line: "Patient:" on one line, name on the next
        for i, lt in enumerate(line_texts):
            if re.match(r"^Patient(?:'s Name)?:?\s*$", lt, re.I) and i + 1 < len(line_texts):
                name = line_texts[i + 1].strip()
                if name and len(name) > 2 and not name.startswith("Date"):
                    result["patient_name"] = name
                    break
        # Try same-line patterns
        if "patient_name" not in result:
            for pat in [
                re.compile(r"Patient(?:'s)?\s*(?:Name)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)", re.I),
            ]:
                m = pat.search(full_text)
                if m:
                    result["patient_name"] = m.group(1).strip()
                    break

    if "patient_dob" not in result:
        # Multi-line: "Date of Birth:" then date on next line
        for i, lt in enumerate(line_texts):
            if re.match(r"^(?:Date of Birth|DOB):?\s*$", lt, re.I) and i + 1 < len(line_texts):
                m = DATE_RE.search(line_texts[i + 1])
                if m:
                    result["patient_dob"] = m.group(1)
                    break
        # Same-line
        if "patient_dob" not in result:
            for pat in [
                re.compile(r"DOB[:\s]+(\d{1,2}/\d{1,2}/\d{4})"),
                re.compile(r"Date of Birth[:\s]+(\d{1,2}/\d{1,2}/\d{4})", re.I),
            ]:
                m = pat.search(full_text)
                if m:
                    result["patient_dob"] = m.group(1)
                    break

    # Promote ICD-10 codes from entities to top level as diagnoses
    if entities.get("icd10_codes"):
        result["diagnoses"] = entities["icd10_codes"]

    # Extract provider from "Provider:" or "Records From:" lines
    if not result.get("provider_name"):
        for i, lt in enumerate(line_texts):
            if re.match(r"^(?:Provider|Records? From|Attending)[:\s]*$", lt, re.I) and i + 1 < len(line_texts):
                prov = line_texts[i + 1].strip()
                if prov and len(prov) > 2:
                    result["provider_name"] = prov
                    break
        # Fallback: first meaningful line
        if not result.get("provider_name"):
            for lt in line_texts[:5]:
                if len(lt) > 5 and not lt.upper().startswith(("PAGE", "DATE", "CONFID", "MEDICAL RECORD")):
                    result["provider_name"] = lt
                    break

    # Extract MRN if found
    if entities.get("mrn"):
        result["mrn"] = entities["mrn"][0]

    # Summary stats
    result["stats"] = {
        "text_lines": len(lines),
        "controls_total": len(controls),
        "controls_selected": len(selected) if selected else 0,
        "key_value_pairs": len(kv_pairs),
        "dates_found": len(dates),
        "tables_found": len(tables),
        "acroform_fields": len(result.get("acroform_fields", [])),
    }

    return result
