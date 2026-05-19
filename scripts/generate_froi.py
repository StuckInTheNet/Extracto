"""Generate synthetic First Report of Injury (FROI) workers comp forms.

FROI is the initial claim form filed when a workplace injury occurs. Every
state has its own variant (WC-1 Florida, DWC-1 California, C-2F New York),
but they share a common structure.

Key structural features:
- Employee info section
- Employer info section
- Injury details (date/time/location)
- **Body parts injured** — grid of 20+ parts with **left/right laterality**
- Nature of injury (single-select)
- Cause of injury (multi-select)
- Treatment received (multi-select)
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

random.seed(2026)

PAGE_W, PAGE_H = letter
MARGIN_L = 36
MARGIN_R = 36

# Body parts: (name, has_laterality)
# When laterality is True, the part has both L and R checkboxes.
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

OCCUPATIONS = [
    "Warehouse Worker", "Construction Laborer", "Truck Driver", "Nurse",
    "Electrician", "Welder", "Office Clerk", "Custodian", "Machine Operator",
    "Carpenter", "Plumber", "Landscaper", "Assembly Line Worker",
]

INDUSTRIES = [
    ("Manufacturing", "31-33"),
    ("Construction", "23"),
    ("Transportation", "48-49"),
    ("Healthcare", "62"),
    ("Retail Trade", "44-45"),
    ("Warehousing", "493"),
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]


def rand_date(start="2022-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    return d.strftime("%m/%d/%Y")


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_time():
    hour = random.randint(1, 12)
    minute = random.randint(0, 59)
    am_pm = random.choice(["AM", "PM"])
    return f"{hour:02d}:{minute:02d} {am_pm}"


# --- Drawing primitives ---

def draw_checkbox(c: canvas.Canvas, x, y, size=9, checked=False, style="check"):
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.6)
    c.rect(x, y, size, size, stroke=1, fill=0)
    if checked:
        if style == "check":
            c.setLineWidth(1.3)
            c.line(x + 1.5, y + size / 2 - 0.5, x + size / 3, y + 1.5)
            c.line(x + size / 3, y + 1.5, x + size - 1.5, y + size - 1.5)
            c.setLineWidth(0.6)
        elif style == "x":
            c.setLineWidth(1.3)
            c.line(x + 1.5, y + 1.5, x + size - 1.5, y + size - 1.5)
            c.line(x + 1.5, y + size - 1.5, x + size - 1.5, y + 1.5)
            c.setLineWidth(0.6)


def draw_underline(c: canvas.Canvas, x, y, w):
    c.setStrokeColor(colors.lightgrey)
    c.line(x, y, x + w, y)
    c.setStrokeColor(colors.black)


def draw_text_field(c: canvas.Canvas, x, y, label, value, label_w=90, field_w=180):
    c.setFont("Helvetica", 8)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica", 9)
    c.drawString(x + label_w, y, str(value))
    draw_underline(c, x + label_w, y - 2, field_w)


# --- Section drawers ---

def draw_header(c: canvas.Canvas, state_code: str):
    c.setFont("Helvetica-Bold", 12)
    title = f"EMPLOYER'S FIRST REPORT OF INJURY (Form {state_code})"
    c.drawCentredString(PAGE_W / 2, PAGE_H - 40, title)
    c.setFont("Helvetica", 7)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 52, "Workers' Compensation Division")


def draw_employee_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "EMPLOYEE INFORMATION")
    y -= 14

    draw_text_field(c, MARGIN_L, y, "Employee Name", data["employee_name"], label_w=85, field_w=200)
    draw_text_field(c, MARGIN_L + 320, y, "SSN (last 4)", data["employee_ssn_last4"], label_w=60, field_w=80)
    y -= 16
    draw_text_field(c, MARGIN_L, y, "Date of Birth", data["employee_dob"], label_w=85, field_w=100)
    draw_text_field(c, MARGIN_L + 320, y, "Hire Date", data["employee_hire_date"], label_w=60, field_w=100)
    y -= 16
    draw_text_field(c, MARGIN_L, y, "Occupation", data["employee_occupation"], label_w=85, field_w=200)
    y -= 16
    draw_text_field(c, MARGIN_L, y, "Address", data["employee_address"], label_w=85, field_w=380)
    return y - 10


def draw_employer_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "EMPLOYER INFORMATION")
    y -= 14

    draw_text_field(c, MARGIN_L, y, "Employer Name", data["employer_name"], label_w=85, field_w=220)
    draw_text_field(c, MARGIN_L + 340, y, "FEIN", data["employer_fein"], label_w=40, field_w=100)
    y -= 16
    draw_text_field(c, MARGIN_L, y, "Industry", data["employer_industry"], label_w=85, field_w=150)
    draw_text_field(c, MARGIN_L + 280, y, "NAICS", data["employer_naics"], label_w=40, field_w=80)
    y -= 16
    draw_text_field(c, MARGIN_L, y, "Address", data["employer_address"], label_w=85, field_w=380)
    return y - 10


def draw_injury_details(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "INJURY DETAILS")
    y -= 14

    draw_text_field(c, MARGIN_L, y, "Date of Injury", data["injury_date"], label_w=85, field_w=100)
    draw_text_field(c, MARGIN_L + 215, y, "Time", data["injury_time"], label_w=30, field_w=100)
    y -= 16

    # On premises Y/N
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L, y, "On employer premises?")
    yes_x = MARGIN_L + 140
    no_x = MARGIN_L + 180
    draw_checkbox(c, yes_x, y - 2, size=9, checked=data["on_premises"], style=data["mark_style"])
    c.drawString(yes_x + 12, y, "Yes")
    draw_checkbox(c, no_x, y - 2, size=9, checked=not data["on_premises"], style=data["mark_style"])
    c.drawString(no_x + 12, y, "No")
    y -= 16

    draw_text_field(c, MARGIN_L, y, "Location", data["injury_location"], label_w=85, field_w=380)
    return y - 10


def draw_body_parts_grid(c: canvas.Canvas, y: float, data: dict) -> float:
    """The distinctive FROI feature: grid of body parts with L/R checkboxes."""
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "BODY PARTS INJURED")
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L + 140, y, "(Check all that apply; L = Left, R = Right)")
    y -= 14

    injured = set(data["injured_body_parts"])

    # Lay out the body parts in 3 columns. Each body part uses a single row
    # with either 1 checkbox (no laterality) or 2 checkboxes (L + R).
    usable_w = PAGE_W - MARGIN_L - MARGIN_R
    col_w = usable_w / 3
    n_rows = (len(BODY_PARTS) + 2) // 3
    row_h = 13

    c.setFont("Helvetica", 8)
    for i, (part, has_lat) in enumerate(BODY_PARTS):
        col = i // n_rows
        row = i % n_rows
        cx0 = MARGIN_L + col * col_w
        cy = y - row * row_h

        # Part name
        c.drawString(cx0, cy, part)

        if has_lat:
            # L checkbox
            l_checked = f"{part}-L" in injured
            l_x = cx0 + 80
            draw_checkbox(c, l_x, cy - 2, size=8, checked=l_checked, style=data["mark_style"])
            c.drawString(l_x + 10, cy, "L")
            # R checkbox
            r_checked = f"{part}-R" in injured
            r_x = cx0 + 110
            draw_checkbox(c, r_x, cy - 2, size=8, checked=r_checked, style=data["mark_style"])
            c.drawString(r_x + 10, cy, "R")
        else:
            # Single checkbox
            checked = part in injured
            single_x = cx0 + 80
            draw_checkbox(c, single_x, cy - 2, size=8, checked=checked, style=data["mark_style"])

    return y - n_rows * row_h - 10


def draw_nature_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "NATURE OF INJURY")
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L + 130, y, "(select one)")
    y -= 14

    # 3-column grid
    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    c.setFont("Helvetica", 9)
    for i, nature in enumerate(NATURE_OF_INJURY):
        col = i % 3
        row = i // 3
        cx = MARGIN_L + col * col_w
        cy = y - row * 14
        checked = nature == data["nature_of_injury"]
        draw_checkbox(c, cx, cy - 2, size=9, checked=checked, style=data["mark_style"])
        c.drawString(cx + 13, cy, nature)
    return y - ((len(NATURE_OF_INJURY) + 2) // 3) * 14 - 8


def draw_cause_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "CAUSE OF INJURY")
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L + 130, y, "(check all that apply)")
    y -= 14

    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    c.setFont("Helvetica", 9)
    causes = set(data["causes"])
    for i, cause in enumerate(CAUSE_OF_INJURY):
        col = i % 3
        row = i // 3
        cx = MARGIN_L + col * col_w
        cy = y - row * 14
        checked = cause in causes
        draw_checkbox(c, cx, cy - 2, size=9, checked=checked, style=data["mark_style"])
        c.drawString(cx + 13, cy, cause)
    return y - ((len(CAUSE_OF_INJURY) + 2) // 3) * 14 - 8


def draw_treatment_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "TREATMENT RECEIVED")
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L + 130, y, "(check all that apply)")
    y -= 14

    col_w = (PAGE_W - MARGIN_L - MARGIN_R) / 3
    c.setFont("Helvetica", 9)
    treatments = set(data["treatments"])
    for i, t in enumerate(TREATMENT_OPTIONS):
        col = i % 3
        row = i // 3
        cx = MARGIN_L + col * col_w
        cy = y - row * 14
        checked = t in treatments
        draw_checkbox(c, cx, cy - 2, size=9, checked=checked, style=data["mark_style"])
        c.drawString(cx + 13, cy, t)
    return y - ((len(TREATMENT_OPTIONS) + 2) // 3) * 14 - 10


# --- Scenario generation ---

def generate_scenario() -> dict[str, Any]:
    n_parts = random.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.18, 0.07])[0]

    injured_parts: list[str] = []
    for _ in range(n_parts):
        part, has_lat = random.choice(BODY_PARTS)
        if has_lat:
            side = random.choice(["L", "R"])
            injured_parts.append(f"{part}-{side}")
        else:
            injured_parts.append(part)
    injured_parts = list(set(injured_parts))  # dedupe

    industry, naics = random.choice(INDUSTRIES)

    n_causes = random.choices([1, 2], weights=[0.7, 0.3])[0]
    causes = random.sample(CAUSE_OF_INJURY, k=n_causes)

    n_treatments = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]
    treatments = random.sample(TREATMENT_OPTIONS, k=n_treatments)

    state = random.choice(["WC-1 FL", "DWC-1 CA", "C-2F NY", "OSHA 301", "Form 45"])

    return {
        "state_code": state,
        "employee_name": rand_name(),
        "employee_ssn_last4": str(random.randint(1000, 9999)),
        "employee_dob": rand_date(start="1960-01-01", end="2003-12-31"),
        "employee_hire_date": rand_date(start="2015-01-01", end="2024-01-01"),
        "employee_occupation": random.choice(OCCUPATIONS),
        "employee_address": f"{random.randint(100, 9999)} Elm St, {random.choice(STATES)}",
        "employer_name": f"{random.choice(['Acme', 'Apex', 'Summit', 'Pioneer'])} {industry} Corp",
        "employer_fein": f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
        "employer_industry": industry,
        "employer_naics": naics,
        "employer_address": f"{random.randint(100, 9999)} Industrial Blvd, {random.choice(STATES)}",
        "injury_date": rand_date(),
        "injury_time": rand_time(),
        "on_premises": random.random() < 0.85,
        "injury_location": random.choice([
            "Warehouse Floor", "Loading Dock", "Break Room", "Production Line",
            "Office", "Parking Lot", "Construction Site",
        ]),
        "injured_body_parts": sorted(injured_parts),
        "nature_of_injury": random.choice(NATURE_OF_INJURY),
        "causes": causes,
        "treatments": treatments,
        "mark_style": random.choice(["check", "x"]),
    }


def generate_form(out_pdf: str, data: dict):
    c = canvas.Canvas(out_pdf, pagesize=letter)
    draw_header(c, data["state_code"])

    y = PAGE_H - 75
    y = draw_employee_section(c, y, data)
    y = draw_employer_section(c, y, data)
    y = draw_injury_details(c, y, data)
    y = draw_body_parts_grid(c, y, data)
    y = draw_nature_section(c, y, data)
    y = draw_cause_section(c, y, data)
    y = draw_treatment_section(c, y, data)

    c.showPage()
    c.save()


def ground_truth(data: dict) -> dict:
    return {
        "form_type": "froi",
        "employee_name": data["employee_name"],
        "employer_name": data["employer_name"],
        "injury_date": data["injury_date"],
        "on_premises": data["on_premises"],
        "injured_body_parts": sorted(data["injured_body_parts"]),
        "nature_of_injury": data["nature_of_injury"],
        "causes": sorted(data["causes"]),
        "treatments": sorted(data["treatments"]),
    }


def generate_dataset(out_dir: str = "dataset/froi", n: int = 25) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"froi": []}
    for i in range(1, n + 1):
        data = generate_scenario()
        pdf_path = out_path / f"froi_{i:03d}.pdf"
        json_path = out_path / f"froi_{i:03d}.json"
        generate_form(str(pdf_path), data)
        json_path.write_text(json.dumps(ground_truth(data), indent=2))
        manifest["froi"].append({"pdf": str(pdf_path), "json": str(json_path)})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    n = int(os.environ.get("N_FORMS", "25"))
    out_dir = os.environ.get("OUT_DIR", "dataset/froi")
    manifest = generate_dataset(out_dir, n)
    print(f"Generated {len(manifest['froi'])} FROI forms in {out_dir}")
