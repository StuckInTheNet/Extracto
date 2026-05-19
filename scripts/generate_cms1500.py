"""Generate synthetic CMS-1500 health insurance claim forms.

CMS-1500 is the standard form used by non-institutional providers to bill
Medicare and other insurance carriers. It's the single most common medical
bill a personal injury attorney will encounter.

This generator produces realistic forms with varied scenarios:
- Different insurance types (Medicare, Medicaid, Group Health, etc.)
- Employment vs auto accident vs other injury origins
- Variable service line counts (1-6 rows in Box 24)
- Multiple diagnosis codes (1-12 in Box 21)
- Different provider IDs and tax ID types

Each PDF is paired with a ground-truth JSON containing the key extractable
fields so we can measure extraction accuracy.
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

random.seed(2024)

PAGE_W, PAGE_H = letter
MARGIN_L = 18
MARGIN_R = 18

# Insurance type options (Box 1)
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

# Common ICD-10 codes for PI/med-mal cases
ICD10_POOL = [
    ("S13.4XXA", "Sprain of ligaments of cervical spine"),
    ("M54.2", "Cervicalgia"),
    ("M54.5", "Low back pain"),
    ("M51.26", "Lumbar disc displacement"),
    ("S22.5XXA", "Fracture of sternum"),
    ("S42.201A", "Fracture of upper end of humerus"),
    ("S83.511A", "Sprain of anterior cruciate ligament"),
    ("G44.311", "Acute post-traumatic headache"),
    ("R51.9", "Headache, unspecified"),
    ("F43.10", "Post-traumatic stress disorder"),
    ("F41.1", "Generalized anxiety disorder"),
    ("F32.9", "Major depressive disorder"),
    ("S06.0X0A", "Concussion without loss of consciousness"),
    ("M25.511", "Pain in right shoulder"),
    ("M25.512", "Pain in left shoulder"),
    ("M79.2", "Neuralgia and neuritis, unspecified"),
    ("M62.830", "Muscle spasm of back"),
    ("R52", "Pain, unspecified"),
    ("S39.012A", "Strain of muscle of lower back"),
    ("T14.90XA", "Injury, unspecified"),
]

# CPT codes for common PI-related services
CPT_POOL = [
    ("99203", "Office visit new patient, moderate", 225.00),
    ("99213", "Office visit established patient", 125.00),
    ("99214", "Office visit established, moderate", 175.00),
    ("97110", "Therapeutic exercise", 85.00),
    ("97140", "Manual therapy", 80.00),
    ("97112", "Neuromuscular reeducation", 90.00),
    ("72148", "MRI lumbar spine without contrast", 1250.00),
    ("72141", "MRI cervical spine without contrast", 1200.00),
    ("73721", "MRI lower extremity joint", 1100.00),
    ("72100", "X-ray lumbar spine", 185.00),
    ("72040", "X-ray cervical spine", 175.00),
    ("20610", "Joint injection, major", 320.00),
    ("64483", "Transforaminal epidural injection", 850.00),
    ("95910", "Nerve conduction study", 475.00),
    ("98941", "Chiropractic manipulation 3-4 regions", 65.00),
]

PLACES_OF_SERVICE = [
    ("11", "Office"),
    ("22", "Outpatient Hospital"),
    ("23", "Emergency Room"),
    ("12", "Home"),
    ("21", "Inpatient Hospital"),
]

PROVIDERS = [
    ("Sarah Mitchell, MD", "1234567890", "12-3456789"),
    ("James Rodriguez, DO", "2345678901", "23-4567890"),
    ("Emily Chen, MD", "3456789012", "34-5678901"),
    ("Michael Thompson, DC", "4567890123", "45-6789012"),
    ("Priya Patel, MD", "5678901234", "56-7890123"),
    ("David Kim, MD", "6789012345", "67-8901234"),
    ("Lisa Washington, PT", "7890123456", "78-9012345"),
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron", "Sam", "Pat", "Robin", "Kelly", "Chris"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson"]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]


def rand_date(start="2023-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    return d.strftime("%m/%d/%Y")


def rand_name():
    return f"{random.choice(LAST_NAMES)}, {random.choice(FIRST_NAMES)}"


# --- Drawing primitives ---

def draw_box(c: canvas.Canvas, x, y, w, h, label=None, label_size=5.5):
    """Draw a numbered form box."""
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(x, y, w, h, stroke=1, fill=0)
    if label:
        c.setFont("Helvetica", label_size)
        c.drawString(x + 1.5, y + h - label_size - 1, label)


def draw_checkbox(c: canvas.Canvas, x, y, size=7, checked=False):
    """Draw a checkbox at (x, y) - bottom-left in reportlab coords."""
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.6)
    c.rect(x, y, size, size, stroke=1, fill=0)
    if checked:
        # Draw an X mark
        c.setLineWidth(0.9)
        c.line(x + 1, y + 1, x + size - 1, y + size - 1)
        c.line(x + 1, y + size - 1, x + size - 1, y + 1)
        c.setLineWidth(0.6)


def draw_text(c: canvas.Canvas, x, y, text, size=8, font="Helvetica"):
    c.setFont(font, size)
    c.drawString(x, y, str(text))


# --- Section drawers ---

def draw_header(c: canvas.Canvas, insurance_type: str):
    """Draw the CMS-1500 header with title and Box 1 (insurance type checkboxes)."""
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, PAGE_H - 28, "HEALTH INSURANCE CLAIM FORM")
    c.setFont("Helvetica", 6)
    c.drawString(MARGIN_L, PAGE_H - 37, "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE (NUCC) 02/12")
    c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 28, "PICA")

    # Box 1: insurance type (7 checkboxes in a row)
    box_y = PAGE_H - 68
    box_h = 22
    box_w = (PAGE_W - MARGIN_L - MARGIN_R) * 0.58
    draw_box(c, MARGIN_L, box_y, box_w, box_h, "1. MEDICARE     MEDICAID     TRICARE     CHAMPVA     GROUP     FECA     OTHER")

    cb_y = box_y + 4
    col_w = box_w / 7
    for i, label in enumerate(INSURANCE_TYPES):
        cx = MARGIN_L + i * col_w + 6
        checked = (label == insurance_type)
        draw_checkbox(c, cx, cb_y, size=7, checked=checked)

    # Box 1a: Insured's ID (right side)
    box_1a_x = MARGIN_L + box_w
    box_1a_w = PAGE_W - MARGIN_R - box_1a_x
    draw_box(c, box_1a_x, box_y, box_1a_w, box_h, "1a. INSURED'S I.D. NUMBER")


def draw_patient_info(c: canvas.Canvas, data: dict) -> dict:
    """Draw boxes 2-11: patient and insured info."""
    y = PAGE_H - 90
    row_h = 22
    col_left_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    col_mid_w = col_left_w
    col_right_w = col_left_w

    # Box 2: Patient name
    draw_box(c, MARGIN_L, y, col_left_w, row_h, "2. PATIENT'S NAME (Last, First, MI)")
    draw_text(c, MARGIN_L + 4, y + 6, data["patient_name"], size=9)

    # Box 3: DOB + Sex
    box3_x = MARGIN_L + col_left_w
    draw_box(c, box3_x, y, col_mid_w, row_h, "3. PATIENT'S BIRTH DATE            SEX")
    draw_text(c, box3_x + 4, y + 6, data["patient_dob"], size=9)
    # Sex checkboxes
    sex_cx_m = box3_x + col_mid_w - 60
    sex_cx_f = box3_x + col_mid_w - 25
    draw_checkbox(c, sex_cx_m, y + 6, size=7, checked=data["patient_sex"] == "M")
    draw_text(c, sex_cx_m + 10, y + 7, "M", size=8)
    draw_checkbox(c, sex_cx_f, y + 6, size=7, checked=data["patient_sex"] == "F")
    draw_text(c, sex_cx_f + 10, y + 7, "F", size=8)

    # Box 4: Insured's name
    box4_x = box3_x + col_mid_w
    draw_box(c, box4_x, y, col_right_w, row_h, "4. INSURED'S NAME (Last, First, MI)")
    draw_text(c, box4_x + 4, y + 6, data["insured_name"], size=9)

    y -= row_h

    # Box 5: Patient address
    draw_box(c, MARGIN_L, y, col_left_w, row_h, "5. PATIENT'S ADDRESS (No., Street)")
    draw_text(c, MARGIN_L + 4, y + 6, data["patient_address"], size=9)

    # Box 6: Patient relationship to insured
    draw_box(c, box3_x, y, col_mid_w, row_h, "6. PATIENT RELATIONSHIP TO INSURED")
    rel_y = y + 6
    for i, rel in enumerate(RELATIONSHIPS):
        rx = box3_x + 8 + i * ((col_mid_w - 16) / 4)
        draw_checkbox(c, rx, rel_y, size=7, checked=(rel == data["relationship"]))
        draw_text(c, rx + 10, rel_y + 1, rel, size=7)

    # Box 7: Insured's address
    draw_box(c, box4_x, y, col_right_w, row_h, "7. INSURED'S ADDRESS (No., Street)")
    draw_text(c, box4_x + 4, y + 6, data["insured_address"], size=9)

    return {"patient_info_bottom_y": y}


def draw_condition_box(c: canvas.Canvas, y: float, data: dict):
    """Draw Box 10: Is patient's condition related to (Employment, Auto, Other)."""
    box_x = MARGIN_L + (PAGE_W - MARGIN_L - MARGIN_R) / 3
    box_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    row_h = 44
    draw_box(c, box_x, y - row_h + 22, box_w, row_h, "10. IS PATIENT'S CONDITION RELATED TO:")

    inner_y = y - 2
    # 10a Employment
    draw_text(c, box_x + 4, inner_y, "a. EMPLOYMENT? (Current or Previous)", size=6)
    yes_x = box_x + box_w - 65
    no_x = box_x + box_w - 30
    draw_checkbox(c, yes_x, inner_y - 2, size=7, checked=data["condition_employment"])
    draw_text(c, yes_x + 10, inner_y - 1, "YES", size=6)
    draw_checkbox(c, no_x, inner_y - 2, size=7, checked=not data["condition_employment"])
    draw_text(c, no_x + 10, inner_y - 1, "NO", size=6)

    # 10b Auto accident
    inner_y -= 14
    draw_text(c, box_x + 4, inner_y, "b. AUTO ACCIDENT?", size=6)
    draw_text(c, box_x + 75, inner_y, f"PLACE (State): {data['auto_state']}", size=6)
    draw_checkbox(c, yes_x, inner_y - 2, size=7, checked=data["condition_auto"])
    draw_text(c, yes_x + 10, inner_y - 1, "YES", size=6)
    draw_checkbox(c, no_x, inner_y - 2, size=7, checked=not data["condition_auto"])
    draw_text(c, no_x + 10, inner_y - 1, "NO", size=6)

    # 10c Other accident
    inner_y -= 14
    draw_text(c, box_x + 4, inner_y, "c. OTHER ACCIDENT?", size=6)
    draw_checkbox(c, yes_x, inner_y - 2, size=7, checked=data["condition_other"])
    draw_text(c, yes_x + 10, inner_y - 1, "YES", size=6)
    draw_checkbox(c, no_x, inner_y - 2, size=7, checked=not data["condition_other"])
    draw_text(c, no_x + 10, inner_y - 1, "NO", size=6)


def draw_diagnosis_codes(c: canvas.Canvas, y: float, codes: list[str]):
    """Draw Box 21: Diagnosis or Nature of Illness (up to 12 codes)."""
    box_x = MARGIN_L
    box_w = PAGE_W - MARGIN_L - MARGIN_R
    row_h = 26
    draw_box(c, box_x, y - row_h, box_w, row_h, "21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY  Relate A-L to service line below (24E)")

    # 12 slots in a 4x3 grid with letters A-L
    cell_w = (box_w - 20) / 4
    cell_h = 10
    labels = "ABCDEFGHIJKL"
    for i in range(12):
        col = i % 4
        row = i // 4
        cx = box_x + 10 + col * cell_w
        cy = y - 14 - row * cell_h
        draw_text(c, cx - 6, cy, f"{labels[i]}.", size=6)
        if i < len(codes):
            draw_text(c, cx, cy, codes[i], size=7)


def draw_service_lines(c: canvas.Canvas, y: float, lines: list[dict]):
    """Draw Box 24: Service lines table (up to 6 rows)."""
    box_x = MARGIN_L
    box_w = PAGE_W - MARGIN_L - MARGIN_R
    header_h = 14
    row_h = 18

    # Column layout (percentages of box_w)
    cols = [
        ("A. DATE(S) OF SERVICE From", 0.10),
        ("To", 0.08),
        ("B. POS", 0.05),
        ("C. EMG", 0.04),
        ("D. CPT/HCPCS", 0.12),
        ("E. DIAG PTR", 0.08),
        ("F. $ CHARGES", 0.12),
        ("G. DAYS", 0.06),
        ("H. FAMILY", 0.05),
        ("I. ID QUAL", 0.06),
        ("J. RENDERING PROVIDER ID#", 0.24),
    ]
    # Compute column x positions
    col_xs = []
    x_cursor = box_x
    for _, w_pct in cols:
        col_xs.append(x_cursor)
        x_cursor += box_w * w_pct
    col_xs.append(x_cursor)

    # Header row
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(box_x, y - header_h, box_w, header_h, stroke=1, fill=0)
    for (label, _), cx in zip(cols, col_xs):
        c.setFont("Helvetica", 4.5)
        c.drawString(cx + 2, y - header_h + 6, label)

    # Data rows
    for row_idx in range(6):
        row_y = y - header_h - (row_idx + 1) * row_h
        c.rect(box_x, row_y, box_w, row_h, stroke=1, fill=0)
        # Draw vertical lines
        for cx in col_xs[1:-1]:
            c.line(cx, row_y, cx, row_y + row_h)

        if row_idx < len(lines):
            line = lines[row_idx]
            c.setFont("Helvetica", 8)
            c.drawString(col_xs[0] + 2, row_y + 6, line["date_from"])
            c.drawString(col_xs[1] + 2, row_y + 6, line["date_to"])
            c.drawString(col_xs[2] + 2, row_y + 6, line["pos"])
            c.drawString(col_xs[3] + 2, row_y + 6, "")
            c.drawString(col_xs[4] + 2, row_y + 6, line["cpt"])
            c.drawString(col_xs[5] + 2, row_y + 6, line["diag_ptr"])
            c.drawRightString(col_xs[7] - 2, row_y + 6, f"${line['charge']:.2f}")
            c.drawString(col_xs[7] + 2, row_y + 6, str(line["units"]))

    return y - header_h - 6 * row_h


def draw_totals(c: canvas.Canvas, y: float, data: dict):
    """Draw Box 25-33: Tax ID, totals, signatures."""
    box_w = (PAGE_W - MARGIN_L - MARGIN_R)
    row_h = 22
    y -= 4

    # Box 25: Federal Tax ID (with SSN/EIN)
    box25_w = box_w * 0.25
    draw_box(c, MARGIN_L, y - row_h, box25_w, row_h, "25. FEDERAL TAX I.D. NUMBER    SSN   EIN")
    draw_text(c, MARGIN_L + 4, y - row_h + 6, data["tax_id"], size=9)
    ssn_x = MARGIN_L + box25_w - 40
    ein_x = MARGIN_L + box25_w - 20
    draw_checkbox(c, ssn_x, y - row_h + 6, size=6, checked=data["tax_id_type"] == "SSN")
    draw_checkbox(c, ein_x, y - row_h + 6, size=6, checked=data["tax_id_type"] == "EIN")

    # Box 26: Patient account number
    box26_x = MARGIN_L + box25_w
    box26_w = box_w * 0.20
    draw_box(c, box26_x, y - row_h, box26_w, row_h, "26. PATIENT'S ACCOUNT NO.")
    draw_text(c, box26_x + 4, y - row_h + 6, data["patient_account"], size=9)

    # Box 27: Accept assignment
    box27_x = box26_x + box26_w
    box27_w = box_w * 0.15
    draw_box(c, box27_x, y - row_h, box27_w, row_h, "27. ACCEPT ASSIGNMENT?")
    assign_y = y - row_h + 6
    draw_checkbox(c, box27_x + 10, assign_y, size=7, checked=data["accept_assignment"])
    draw_text(c, box27_x + 22, assign_y + 1, "YES", size=7)
    draw_checkbox(c, box27_x + 45, assign_y, size=7, checked=not data["accept_assignment"])
    draw_text(c, box27_x + 57, assign_y + 1, "NO", size=7)

    # Box 28: Total charge
    box28_x = box27_x + box27_w
    box28_w = box_w * 0.13
    draw_box(c, box28_x, y - row_h, box28_w, row_h, "28. TOTAL CHARGE")
    draw_text(c, box28_x + 4, y - row_h + 6, f"${data['total_charge']:.2f}", size=9)

    # Box 29: Amount paid
    box29_x = box28_x + box28_w
    box29_w = box_w * 0.12
    draw_box(c, box29_x, y - row_h, box29_w, row_h, "29. AMOUNT PAID")
    draw_text(c, box29_x + 4, y - row_h + 6, f"${data['amount_paid']:.2f}", size=9)

    # Box 30: Reserved for NUCC
    box30_x = box29_x + box29_w
    box30_w = box_w - (box30_x - MARGIN_L)
    draw_box(c, box30_x, y - row_h, box30_w, row_h, "30. Rsvd for NUCC Use")

    return y - row_h


def draw_provider_box(c: canvas.Canvas, y: float, data: dict):
    """Draw Box 31-33: Provider signature and billing info."""
    box_w = PAGE_W - MARGIN_L - MARGIN_R
    row_h = 30
    y -= 2

    # Box 31: Physician signature
    b31_w = box_w * 0.35
    draw_box(c, MARGIN_L, y - row_h, b31_w, row_h, "31. SIGNATURE OF PHYSICIAN OR SUPPLIER")
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(MARGIN_L + 6, y - row_h + 10, data["provider_name"])
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L + 6, y - row_h + 2, f"DATE: {data['signature_date']}")

    # Box 32: Service facility
    b32_x = MARGIN_L + b31_w
    b32_w = box_w * 0.35
    draw_box(c, b32_x, y - row_h, b32_w, row_h, "32. SERVICE FACILITY LOCATION")
    draw_text(c, b32_x + 4, y - row_h + 10, data["facility_name"], size=8)
    draw_text(c, b32_x + 4, y - row_h + 2, data["facility_address"], size=7)

    # Box 33: Billing provider info & NPI
    b33_x = b32_x + b32_w
    b33_w = box_w - (b33_x - MARGIN_L)
    draw_box(c, b33_x, y - row_h, b33_w, row_h, "33. BILLING PROVIDER INFO & PH #")
    draw_text(c, b33_x + 4, y - row_h + 10, data["provider_name"], size=8)
    draw_text(c, b33_x + 4, y - row_h + 2, f"NPI: {data['provider_npi']}", size=8)


# --- Scenario generation ---

def generate_scenario() -> dict[str, Any]:
    """Generate realistic random data for a CMS-1500 form."""
    insurance = random.choice(INSURANCE_TYPES)

    # Condition related to - at least one should be true for a PI case
    scenario_type = random.choices(
        ["auto", "work", "other", "none"],
        weights=[0.40, 0.25, 0.15, 0.20],
    )[0]

    condition_employment = scenario_type == "work"
    condition_auto = scenario_type == "auto"
    condition_other = scenario_type == "other"

    # Generate service lines
    n_services = random.randint(1, 6)
    service_date = rand_date(start="2024-01-01")
    lines = []
    total = 0.0
    for _ in range(n_services):
        cpt_code, _, base_charge = random.choice(CPT_POOL)
        pos_code, _ = random.choice(PLACES_OF_SERVICE)
        units = random.randint(1, 3)
        line_charge = base_charge * units
        lines.append({
            "date_from": service_date,
            "date_to": service_date,
            "pos": pos_code,
            "cpt": cpt_code,
            "diag_ptr": random.choice(["A", "AB", "A,B", "ABC"]),
            "charge": line_charge,
            "units": units,
        })
        total += line_charge

    # Diagnosis codes
    n_dx = random.randint(1, 6)
    diagnoses = random.sample(ICD10_POOL, k=n_dx)
    dx_codes = [d[0] for d in diagnoses]

    provider_name, provider_npi, tax_id = random.choice(PROVIDERS)
    tax_id_type = random.choice(["SSN", "EIN"])

    patient = rand_name()

    return {
        "insurance_type": insurance,
        "patient_name": patient,
        "patient_dob": rand_date(start="1950-01-01", end="2005-12-31"),
        "patient_sex": random.choice(["M", "F"]),
        "patient_address": f"{random.randint(100, 9999)} Main St, {random.choice(STATES)}",
        "relationship": random.choices(RELATIONSHIPS, weights=[0.55, 0.25, 0.15, 0.05])[0],
        "insured_name": patient if random.random() < 0.6 else rand_name(),
        "insured_address": f"{random.randint(100, 9999)} Oak Ave, {random.choice(STATES)}",
        "condition_employment": condition_employment,
        "condition_auto": condition_auto,
        "condition_other": condition_other,
        "auto_state": random.choice(STATES) if condition_auto else "",
        "diagnoses": dx_codes,
        "diagnosis_descriptions": [d[1] for d in diagnoses],
        "service_lines": lines,
        "total_charge": total,
        "amount_paid": round(total * random.uniform(0.0, 0.5), 2) if random.random() < 0.3 else 0.0,
        "tax_id": tax_id,
        "tax_id_type": tax_id_type,
        "patient_account": f"ACC{random.randint(10000, 99999)}",
        "accept_assignment": random.random() < 0.85,
        "provider_name": provider_name,
        "provider_npi": provider_npi,
        "facility_name": f"{random.choice(['Valley', 'Westside', 'Central', 'Oakwood'])} Medical {random.choice(['Center', 'Clinic', 'Associates'])}",
        "facility_address": f"{random.randint(100, 9999)} Medical Dr",
        "signature_date": service_date,
    }


def generate_form(out_pdf: str, data: dict):
    """Render a complete CMS-1500 PDF for the given scenario."""
    c = canvas.Canvas(out_pdf, pagesize=letter)

    draw_header(c, data["insurance_type"])
    draw_patient_info(c, data)

    # Condition box (Box 10)
    condition_y = PAGE_H - 90 - 44
    draw_condition_box(c, condition_y, data)

    # Diagnosis codes (Box 21)
    dx_y = PAGE_H - 300
    draw_diagnosis_codes(c, dx_y, data["diagnoses"])

    # Service lines (Box 24)
    svc_y = dx_y - 40
    svc_bottom = draw_service_lines(c, svc_y, data["service_lines"])

    # Totals (Box 25-30)
    totals_bottom = draw_totals(c, svc_bottom, data)

    # Provider box (Box 31-33)
    draw_provider_box(c, totals_bottom, data)

    c.showPage()
    c.save()


def ground_truth(data: dict) -> dict:
    """Extract the ground-truth fields we expect to measure."""
    return {
        "form_type": "cms1500",
        "insurance_type": data["insurance_type"],
        "patient_name": data["patient_name"],
        "patient_dob": data["patient_dob"],
        "patient_sex": data["patient_sex"],
        "relationship_to_insured": data["relationship"],
        "condition_employment": data["condition_employment"],
        "condition_auto": data["condition_auto"],
        "condition_other": data["condition_other"],
        "auto_state": data["auto_state"],
        "diagnoses": data["diagnoses"],
        "service_lines": [
            {
                "date_from": line["date_from"],
                "cpt": line["cpt"],
                "pos": line["pos"],
                "charge": line["charge"],
                "units": line["units"],
            }
            for line in data["service_lines"]
        ],
        "total_charge": data["total_charge"],
        "accept_assignment": data["accept_assignment"],
        "tax_id": data["tax_id"],
        "tax_id_type": data["tax_id_type"],
        "provider_name": data["provider_name"],
        "provider_npi": data["provider_npi"],
    }


def generate_dataset(out_dir: str = "dataset/cms1500", n: int = 25) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"cms1500": []}
    for i in range(1, n + 1):
        data = generate_scenario()
        pdf_path = out_path / f"cms1500_{i:03d}.pdf"
        json_path = out_path / f"cms1500_{i:03d}.json"
        generate_form(str(pdf_path), data)
        json_path.write_text(json.dumps(ground_truth(data), indent=2))
        manifest["cms1500"].append({"pdf": str(pdf_path), "json": str(json_path)})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    n = int(os.environ.get("N_FORMS", "25"))
    out_dir = os.environ.get("OUT_DIR", "dataset/cms1500")
    manifest = generate_dataset(out_dir, n)
    print(f"Generated {len(manifest['cms1500'])} CMS-1500 forms in {out_dir}")
