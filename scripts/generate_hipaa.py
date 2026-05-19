"""Generate synthetic HIPAA Authorization for Release of PHI forms.

HIPAA authorization is required before a provider will release medical records
to a third party (e.g., a personal injury attorney). Key structural feature:

**Opt-out semantics** for sensitive categories. The patient checks categories
they DO NOT want disclosed. Mental health, HIV/AIDS, substance abuse, and
genetic information each require a separate opt-out — default is to include.

This inverts the usual checkbox interpretation (checked = included).
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

random.seed(2025)

PAGE_W, PAGE_H = letter
MARGIN_L = 54
MARGIN_R = 54

# Sensitive categories that require explicit opt-out (32 CFR Part 2, etc.)
SENSITIVE_CATEGORIES = [
    "Mental Health",
    "HIV/AIDS",
    "Substance Abuse",
    "Genetic Information",
]

# Purpose of disclosure options (one or more can be checked)
PURPOSE_OPTIONS = [
    "Legal Proceedings",
    "Insurance Claim",
    "Personal Use",
    "Continuity of Care",
    "Other",
]

# Record types to release
RECORD_TYPES = [
    "Office Notes",
    "Imaging Reports",
    "Lab Results",
    "Operative Reports",
    "Billing Records",
    "Discharge Summaries",
]

TITLES = [
    "Authorization for Release of Protected Health Information",
    "HIPAA Authorization for Use and Disclosure of PHI",
    "Authorization to Release Medical Records (HIPAA Compliant)",
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]


def rand_date(start="2023-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    return d.strftime("%m/%d/%Y")


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# --- Drawing primitives ---

def draw_checkbox(c: canvas.Canvas, x, y, size=10, checked=False, style="check"):
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.7)
    c.rect(x, y, size, size, stroke=1, fill=0)
    if checked:
        if style == "check":
            c.setLineWidth(1.4)
            c.line(x + 1.5, y + size / 2 - 0.5, x + size / 3, y + 1.5)
            c.line(x + size / 3, y + 1.5, x + size - 1.5, y + size - 1.5)
            c.setLineWidth(0.7)
        elif style == "x":
            c.setLineWidth(1.4)
            c.line(x + 1.5, y + 1.5, x + size - 1.5, y + size - 1.5)
            c.line(x + 1.5, y + size - 1.5, x + size - 1.5, y + 1.5)
            c.setLineWidth(0.7)


def draw_underline(c: canvas.Canvas, x, y, w):
    c.setStrokeColor(colors.lightgrey)
    c.line(x, y, x + w, y)
    c.setStrokeColor(colors.black)


def draw_text_field(c: canvas.Canvas, x, y, label, value, label_w=120, field_w=250):
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica", 10)
    c.drawString(x + label_w, y, str(value))
    draw_underline(c, x + label_w, y - 2, field_w)


# --- Section drawers ---

def draw_header(c: canvas.Canvas, title: str, facility: str):
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 54, title)

    c.setFont("Helvetica", 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 70, facility)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, PAGE_H - 78, PAGE_W - MARGIN_R, PAGE_H - 78)


def draw_patient_section(c: canvas.Canvas, y: float, data: dict):
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "PATIENT INFORMATION")
    y -= 18

    draw_text_field(c, MARGIN_L, y, "Patient Name", data["patient_name"], label_w=90, field_w=250)
    y -= 20
    draw_text_field(c, MARGIN_L, y, "Date of Birth", data["patient_dob"], label_w=90, field_w=120)
    draw_text_field(c, MARGIN_L + 280, y, "MRN", data["mrn"], label_w=40, field_w=150)
    y -= 20
    draw_text_field(c, MARGIN_L, y, "Address", data["patient_address"], label_w=90, field_w=370)
    y -= 20
    draw_text_field(c, MARGIN_L, y, "Phone", data["patient_phone"], label_w=90, field_w=150)
    return y - 8


def draw_records_section(c: canvas.Canvas, y: float, data: dict):
    """Section 2: What records to release and for what dates."""
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "RECORDS TO BE RELEASED")
    y -= 18

    draw_text_field(c, MARGIN_L, y, "Date of Service From", data["date_range_from"], label_w=130, field_w=100)
    draw_text_field(c, MARGIN_L + 260, y, "To", data["date_range_to"], label_w=20, field_w=100)
    y -= 24

    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L, y, "Types of records (check all that apply):")
    y -= 14

    # Grid of record type checkboxes - 3 columns
    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    for i, rec_type in enumerate(RECORD_TYPES):
        col = i % 3
        row = i // 3
        cx = MARGIN_L + col * col_w
        cy = y - row * 18
        checked = rec_type in data["record_types"]
        draw_checkbox(c, cx, cy - 2, size=9, checked=checked, style=data["mark_style"])
        c.setFont("Helvetica", 9)
        c.drawString(cx + 14, cy, rec_type)

    return y - (len(RECORD_TYPES) // 3 + 1) * 18 - 6


def draw_sensitive_section(c: canvas.Canvas, y: float, data: dict):
    """Section 3: OPT-OUT for sensitive categories.

    This is the critical opt-out section. Text explicitly states that checking
    a box means the category will NOT be released.
    """
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "SPECIALLY PROTECTED INFORMATION")
    y -= 15

    c.setFont("Helvetica-Bold", 9)
    prompt = (
        "The records I am authorizing to be released MAY include information about the "
        "following specially protected conditions. If you DO NOT want any of these categories "
        "disclosed, check the corresponding box. Unchecked categories WILL be released."
    )
    _draw_wrapped(c, MARGIN_L, y, prompt, PAGE_W - MARGIN_L - MARGIN_R, font_size=8)
    y -= 32

    # 4 opt-out checkboxes arranged in 2 columns
    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 2
    for i, cat in enumerate(SENSITIVE_CATEGORIES):
        col = i % 2
        row = i // 2
        cx = MARGIN_L + col * col_w
        cy = y - row * 18
        excluded = cat in data["excluded_categories"]
        draw_checkbox(c, cx, cy - 2, size=10, checked=excluded, style=data["mark_style"])
        c.setFont("Helvetica", 9)
        c.drawString(cx + 15, cy, f"Do NOT release {cat}")

    return y - 40


def draw_purpose_section(c: canvas.Canvas, y: float, data: dict):
    """Section 4: Purpose of disclosure checkboxes."""
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "PURPOSE OF DISCLOSURE")
    y -= 18

    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L, y, "Check all that apply:")
    y -= 14

    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    for i, purpose in enumerate(PURPOSE_OPTIONS):
        col = i % 3
        row = i // 3
        cx = MARGIN_L + col * col_w
        cy = y - row * 18
        checked = purpose in data["purposes"]
        draw_checkbox(c, cx, cy - 2, size=9, checked=checked, style=data["mark_style"])
        c.drawString(cx + 14, cy, purpose)

    rows = (len(PURPOSE_OPTIONS) + 2) // 3
    return y - rows * 18 - 6


def draw_recipient_section(c: canvas.Canvas, y: float, data: dict):
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "RELEASE RECORDS TO")
    y -= 18

    draw_text_field(c, MARGIN_L, y, "Name", data["recipient_name"], label_w=60, field_w=300)
    y -= 20
    draw_text_field(c, MARGIN_L, y, "Address", data["recipient_address"], label_w=60, field_w=400)
    y -= 20
    draw_text_field(c, MARGIN_L, y, "Fax", data["recipient_fax"], label_w=60, field_w=150)
    return y - 8


def draw_expiration_section(c: canvas.Canvas, y: float, data: dict):
    """Expiration: one-year default OR specific date."""
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "EXPIRATION")
    y -= 18

    # Two mutually exclusive options - drawn as checkboxes
    c.setFont("Helvetica", 9)
    one_year_checked = data["expiration_type"] == "one_year"
    specific_checked = data["expiration_type"] == "specific_date"

    draw_checkbox(c, MARGIN_L, y - 2, size=9, checked=one_year_checked, style=data["mark_style"])
    c.drawString(MARGIN_L + 14, y, "One year from date of signature")

    y -= 18
    draw_checkbox(c, MARGIN_L, y - 2, size=9, checked=specific_checked, style=data["mark_style"])
    c.drawString(MARGIN_L + 14, y, f"Specific date: {data['expiration_date'] if specific_checked else '_____________'}")

    return y - 8


def draw_signature_section(c: canvas.Canvas, y: float, data: dict):
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "PATIENT SIGNATURE")
    y -= 20

    c.setFont("Helvetica-Oblique", 12)
    c.drawString(MARGIN_L, y, data["patient_name"])
    draw_underline(c, MARGIN_L, y - 2, 200)
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L, y - 14, "Signature of Patient")

    c.setFont("Helvetica", 10)
    c.drawString(PAGE_W - MARGIN_R - 180, y, data["signature_date"])
    draw_underline(c, PAGE_W - MARGIN_R - 180, y - 2, 150)
    c.setFont("Helvetica", 9)
    c.drawString(PAGE_W - MARGIN_R - 180, y - 14, "Date")


def _draw_wrapped(c: canvas.Canvas, x: float, y: float, text: str, max_w: float, font_size=9):
    c.setFont("Helvetica", font_size)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = current + (" " if current else "") + word
        if c.stringWidth(test, "Helvetica", font_size) <= max_w:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        c.drawString(x, y - i * (font_size + 2), line)


# --- Scenario generation ---

def generate_scenario() -> dict[str, Any]:
    # How many sensitive categories are being opted out
    n_excluded = random.choices([0, 1, 2, 3, 4], weights=[0.55, 0.25, 0.12, 0.05, 0.03])[0]
    excluded = random.sample(SENSITIVE_CATEGORIES, k=n_excluded)

    # Purposes (usually 1-2)
    n_purposes = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
    purposes = random.sample(PURPOSE_OPTIONS, k=n_purposes)

    # Record types
    n_records = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.20, 0.30, 0.25, 0.15])[0]
    records = random.sample(RECORD_TYPES, k=n_records)

    expiration_type = random.choices(["one_year", "specific_date"], weights=[0.65, 0.35])[0]

    patient = rand_name()
    date_from = rand_date(start="2022-01-01", end="2023-06-01")
    date_to = rand_date(start="2023-06-02", end="2024-12-31")

    return {
        "patient_name": patient,
        "patient_dob": rand_date(start="1950-01-01", end="2005-12-31"),
        "mrn": f"MRN{random.randint(100000, 999999)}",
        "patient_address": f"{random.randint(100, 9999)} Main St, {random.choice(STATES)}",
        "patient_phone": f"({random.randint(200, 989)}) {random.randint(200, 989)}-{random.randint(1000, 9999)}",
        "date_range_from": date_from,
        "date_range_to": date_to,
        "record_types": records,
        "excluded_categories": excluded,
        "purposes": purposes,
        "recipient_name": f"{random.choice(['Morgan', 'Chen', 'Patel', 'Singh'])} Law Firm, PLLC",
        "recipient_address": f"{random.randint(100, 9999)} Legal Plaza, Suite {random.randint(100, 999)}",
        "recipient_fax": f"({random.randint(200, 989)}) {random.randint(200, 989)}-{random.randint(1000, 9999)}",
        "expiration_type": expiration_type,
        "expiration_date": rand_date(start="2025-01-01", end="2026-12-31") if expiration_type == "specific_date" else "",
        "signature_date": rand_date(start="2024-06-01"),
        "facility": random.choice(["Valley Medical Center", "Westside Health Associates", "Central City Hospital"]),
        "title": random.choice(TITLES),
        "mark_style": random.choice(["check", "x"]),
    }


def generate_form(out_pdf: str, data: dict):
    c = canvas.Canvas(out_pdf, pagesize=letter)

    draw_header(c, data["title"], data["facility"])

    y = PAGE_H - 100
    y = draw_patient_section(c, y, data)
    y = draw_records_section(c, y, data)
    y = draw_sensitive_section(c, y, data)
    y = draw_purpose_section(c, y, data)
    y = draw_recipient_section(c, y, data)
    y = draw_expiration_section(c, y, data)
    draw_signature_section(c, y - 10, data)

    c.showPage()
    c.save()


def ground_truth(data: dict) -> dict:
    return {
        "form_type": "hipaa",
        "patient_name": data["patient_name"],
        "patient_dob": data["patient_dob"],
        "date_range_from": data["date_range_from"],
        "date_range_to": data["date_range_to"],
        "record_types": sorted(data["record_types"]),
        "excluded_categories": sorted(data["excluded_categories"]),
        "purposes": sorted(data["purposes"]),
        "expiration_type": data["expiration_type"],
    }


def generate_dataset(out_dir: str = "dataset/hipaa", n: int = 25) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"hipaa": []}
    for i in range(1, n + 1):
        data = generate_scenario()
        pdf_path = out_path / f"hipaa_{i:03d}.pdf"
        json_path = out_path / f"hipaa_{i:03d}.json"
        generate_form(str(pdf_path), data)
        json_path.write_text(json.dumps(ground_truth(data), indent=2))
        manifest["hipaa"].append({"pdf": str(pdf_path), "json": str(json_path)})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    n = int(os.environ.get("N_FORMS", "25"))
    out_dir = os.environ.get("OUT_DIR", "dataset/hipaa")
    manifest = generate_dataset(out_dir, n)
    print(f"Generated {len(manifest['hipaa'])} HIPAA forms in {out_dir}")
