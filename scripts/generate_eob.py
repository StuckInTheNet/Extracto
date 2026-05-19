"""Generate synthetic Explanation of Benefits (EOB) forms.

EOBs are sent by health insurance payers to members after a claim is processed.
They document what the provider billed, what the insurer allowed, what the
insurer paid, and what the patient owes. For personal injury cases, EOBs are
critical evidence for:
- Calculating collateral source offsets (what insurance actually paid)
- Subrogation / lien resolution
- Showing what's still owed by the patient (damages)

Structural features (the new primitives this adds):
1. Wide financial table with 8-10 columns
2. Reason codes (CARC/RARC) per line item that reference a legend
3. Legend / key at the bottom mapping codes to descriptions
4. Totals row summing each column
5. Patient responsibility summary
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

random.seed(2027)

PAGE_W, PAGE_H = letter
MARGIN_L = 36
MARGIN_R = 36

# CARC (Claim Adjustment Reason Codes) used in real EOBs
REASON_CODES = [
    ("CO-45", "Charge exceeds fee schedule/maximum allowable amount"),
    ("PR-1", "Deductible amount"),
    ("PR-2", "Coinsurance amount"),
    ("PR-3", "Copayment amount"),
    ("CO-96", "Non-covered charge(s)"),
    ("CO-97", "Service included in payment for another service"),
    ("OA-23", "Impact of prior payer(s) adjudication"),
    ("CO-42", "Charges exceed fee schedule"),
    ("PR-27", "Expenses incurred after coverage terminated"),
    ("CO-18", "Duplicate claim/service"),
    ("CO-29", "Time limit for filing has expired"),
    ("CO-119", "Benefit maximum has been reached"),
]

CPT_CODES = [
    ("99203", "Office visit new patient, moderate", 225.00),
    ("99213", "Office visit established patient", 125.00),
    ("99214", "Office visit established, moderate", 175.00),
    ("97110", "Therapeutic exercise", 85.00),
    ("97140", "Manual therapy", 80.00),
    ("72148", "MRI lumbar spine without contrast", 1250.00),
    ("72141", "MRI cervical spine without contrast", 1200.00),
    ("72100", "X-ray lumbar spine", 185.00),
    ("20610", "Joint injection, major", 320.00),
    ("64483", "Transforaminal epidural injection", 850.00),
    ("95910", "Nerve conduction study", 475.00),
    ("98941", "Chiropractic manipulation 3-4 regions", 65.00),
]

PAYERS = [
    "Blue Cross Blue Shield",
    "Aetna Health Insurance",
    "United Healthcare",
    "Cigna HealthCare",
    "Humana Insurance",
    "Kaiser Permanente",
    "Anthem Health",
]

PROVIDERS = [
    ("Sarah Mitchell, MD", "1234567890"),
    ("James Rodriguez, DO", "2345678901"),
    ("Emily Chen, MD", "3456789012"),
    ("Michael Thompson, DC", "4567890123"),
    ("Priya Patel, MD", "5678901234"),
    ("David Kim, MD", "6789012345"),
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]


def rand_date(start="2024-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    return d.strftime("%m/%d/%Y")


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# --- Drawing primitives ---

def draw_underline(c: canvas.Canvas, x, y, w):
    c.setStrokeColor(colors.lightgrey)
    c.line(x, y, x + w, y)
    c.setStrokeColor(colors.black)


def draw_text_field(c: canvas.Canvas, x, y, label, value, label_w=100, field_w=200, size=9):
    c.setFont("Helvetica", size)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica-Bold", size)
    c.drawString(x + label_w, y, str(value))


# --- Sections ---

def draw_header(c: canvas.Canvas, data: dict):
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 40, "EXPLANATION OF BENEFITS")
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 53, "THIS IS NOT A BILL")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN_L, PAGE_H - 75, data["payer_name"])
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN_L, PAGE_H - 86, f"{random.randint(100, 9999)} Insurance Way")
    c.drawString(MARGIN_L, PAGE_H - 96, f"Member Services: 1-800-{random.randint(200,999)}-{random.randint(1000,9999)}")

    # Right side: claim / check info
    right_x = PAGE_W - MARGIN_R - 200
    c.setFont("Helvetica", 9)
    draw_text_field(c, right_x, PAGE_H - 75, "Claim Number", data["claim_number"], label_w=85, field_w=120)
    draw_text_field(c, right_x, PAGE_H - 88, "Check Number", data["check_number"], label_w=85, field_w=120)
    draw_text_field(c, right_x, PAGE_H - 101, "Check Date", data["check_date"], label_w=85, field_w=120)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, PAGE_H - 110, PAGE_W - MARGIN_R, PAGE_H - 110)


def draw_member_provider_section(c: canvas.Canvas, y: float, data: dict) -> float:
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "MEMBER INFORMATION")
    c.drawString(PAGE_W / 2, y, "PROVIDER INFORMATION")
    y -= 16

    draw_text_field(c, MARGIN_L, y, "Member Name", data["member_name"], label_w=85, field_w=180)
    draw_text_field(c, PAGE_W / 2, y, "Provider Name", data["provider_name"], label_w=85, field_w=180)
    y -= 14
    draw_text_field(c, MARGIN_L, y, "Member ID", data["member_id"], label_w=85, field_w=180)
    draw_text_field(c, PAGE_W / 2, y, "Provider NPI", data["provider_npi"], label_w=85, field_w=180)
    y -= 14
    draw_text_field(c, MARGIN_L, y, "Group Number", data["group_number"], label_w=85, field_w=180)
    return y - 12


def draw_service_table(c: canvas.Canvas, y: float, data: dict) -> float:
    """Render the multi-column service line table - the main EOB content."""
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, "CLAIM DETAIL")
    y -= 16

    # Column headers and widths
    columns = [
        ("Date of Service", 0.10),
        ("CPT", 0.07),
        ("Billed", 0.10),
        ("Allowed", 0.10),
        ("Deductible", 0.10),
        ("Copay", 0.08),
        ("Coins.", 0.08),
        ("Plan Paid", 0.10),
        ("Pt Resp", 0.10),
        ("Rsn", 0.17),
    ]
    usable_w = PAGE_W - MARGIN_L - MARGIN_R

    col_xs = [MARGIN_L]
    for _, w_pct in columns:
        col_xs.append(col_xs[-1] + usable_w * w_pct)

    # Header row
    row_h = 16
    c.setFillColor(colors.lightgrey)
    c.rect(MARGIN_L, y - row_h + 4, usable_w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(MARGIN_L, y - row_h + 4, usable_w, row_h, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 8)
    for (label, _), cx in zip(columns, col_xs[:-1]):
        c.drawString(cx + 2, y - 8, label)
    # Vertical column dividers
    for cx in col_xs[1:-1]:
        c.line(cx, y - row_h + 4, cx, y + 4)

    y -= row_h

    # Data rows
    c.setFont("Helvetica", 8)
    for line in data["service_lines"]:
        c.rect(MARGIN_L, y - row_h + 4, usable_w, row_h, stroke=1, fill=0)
        for cx in col_xs[1:-1]:
            c.line(cx, y - row_h + 4, cx, y + 4)

        values = [
            line["date_of_service"],
            line["cpt"],
            f"${line['billed']:.2f}",
            f"${line['allowed']:.2f}",
            f"${line['deductible']:.2f}",
            f"${line['copay']:.2f}",
            f"${line['coinsurance']:.2f}",
            f"${line['plan_paid']:.2f}",
            f"${line['patient_resp']:.2f}",
            ", ".join(line["reason_codes"]),
        ]
        for value, cx in zip(values, col_xs[:-1]):
            c.drawString(cx + 2, y - 8, value)

        y -= row_h

    # Totals row
    c.setFillColor(colors.lightgrey)
    c.rect(MARGIN_L, y - row_h + 4, usable_w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.rect(MARGIN_L, y - row_h + 4, usable_w, row_h, stroke=1, fill=0)
    for cx in col_xs[1:-1]:
        c.line(cx, y - row_h + 4, cx, y + 4)

    c.setFont("Helvetica-Bold", 8)
    c.drawString(col_xs[0] + 2, y - 8, "TOTALS")
    totals = [
        "",
        "",
        f"${data['total_billed']:.2f}",
        f"${data['total_allowed']:.2f}",
        f"${data['total_deductible']:.2f}",
        f"${data['total_copay']:.2f}",
        f"${data['total_coinsurance']:.2f}",
        f"${data['total_plan_paid']:.2f}",
        f"${data['total_patient_resp']:.2f}",
        "",
    ]
    for value, cx in zip(totals[2:], col_xs[2:-1]):
        c.drawString(cx + 2, y - 8, value)

    y -= row_h + 8
    return y


def draw_reason_legend(c: canvas.Canvas, y: float, data: dict) -> float:
    """Print the reason code legend for any codes used in the service lines."""
    used_codes = set()
    for line in data["service_lines"]:
        used_codes.update(line["reason_codes"])
    if not used_codes:
        return y

    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN_L, y, "REASON CODE LEGEND")
    y -= 14

    c.setFont("Helvetica", 8)
    code_map = dict(REASON_CODES)
    for code in sorted(used_codes):
        desc = code_map.get(code, "Unknown")
        c.setFont("Helvetica-Bold", 8)
        c.drawString(MARGIN_L, y, code)
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN_L + 40, y, desc)
        y -= 11

    return y - 8


def draw_summary_box(c: canvas.Canvas, y: float, data: dict):
    """Patient responsibility summary box."""
    box_w = 260
    box_h = 70
    box_x = PAGE_W - MARGIN_R - box_w
    box_y = y - box_h

    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(box_x + 8, y - 14, "PATIENT RESPONSIBILITY SUMMARY")
    c.setFont("Helvetica", 9)

    line_y = y - 28
    rows = [
        ("Deductible:", data["total_deductible"]),
        ("Copay:", data["total_copay"]),
        ("Coinsurance:", data["total_coinsurance"]),
        ("Total Patient Owes:", data["total_patient_resp"]),
    ]
    for label, amount in rows:
        if "Total" in label:
            c.setFont("Helvetica-Bold", 9)
        c.drawString(box_x + 10, line_y, label)
        c.drawRightString(box_x + box_w - 10, line_y, f"${amount:.2f}")
        c.setFont("Helvetica", 9)
        line_y -= 12


# --- Scenario generation ---

def generate_service_line(dos: str) -> dict[str, Any]:
    cpt, _, base = random.choice(CPT_CODES)
    billed = base * random.choice([1.0, 1.1, 1.2, 0.95])
    billed = round(billed, 2)

    # Insurance-typical adjustments
    allowed = round(billed * random.uniform(0.45, 0.80), 2)
    deductible_hit = random.random() < 0.25
    copay_hit = random.random() < 0.30
    coinsurance_hit = random.random() < 0.50

    deductible = round(random.uniform(20, 150), 2) if deductible_hit else 0.0
    copay = round(random.choice([10, 20, 30, 40, 50]), 2) if copay_hit else 0.0

    after_ded_copay = max(0.0, allowed - deductible - copay)
    coinsurance = round(after_ded_copay * 0.2, 2) if coinsurance_hit else 0.0

    plan_paid = round(max(0.0, after_ded_copay - coinsurance), 2)
    patient_resp = round(deductible + copay + coinsurance, 2)

    # Pick applicable reason codes
    reason_codes = []
    if allowed < billed:
        reason_codes.append("CO-45")
    if deductible > 0:
        reason_codes.append("PR-1")
    if copay > 0:
        reason_codes.append("PR-3")
    if coinsurance > 0:
        reason_codes.append("PR-2")

    return {
        "date_of_service": dos,
        "cpt": cpt,
        "billed": billed,
        "allowed": allowed,
        "deductible": deductible,
        "copay": copay,
        "coinsurance": coinsurance,
        "plan_paid": plan_paid,
        "patient_resp": patient_resp,
        "reason_codes": reason_codes,
    }


def generate_scenario() -> dict[str, Any]:
    n_lines = random.randint(2, 6)
    base_date = rand_date(start="2024-01-01", end="2024-11-30")
    service_lines = [generate_service_line(base_date) for _ in range(n_lines)]

    total_billed = round(sum(l["billed"] for l in service_lines), 2)
    total_allowed = round(sum(l["allowed"] for l in service_lines), 2)
    total_deductible = round(sum(l["deductible"] for l in service_lines), 2)
    total_copay = round(sum(l["copay"] for l in service_lines), 2)
    total_coinsurance = round(sum(l["coinsurance"] for l in service_lines), 2)
    total_plan_paid = round(sum(l["plan_paid"] for l in service_lines), 2)
    total_patient_resp = round(sum(l["patient_resp"] for l in service_lines), 2)

    provider_name, provider_npi = random.choice(PROVIDERS)

    return {
        "payer_name": random.choice(PAYERS),
        "member_name": rand_name(),
        "member_id": f"{random.choice(['ABC', 'XYZ', 'DEF'])}{random.randint(100000, 999999)}",
        "group_number": f"GRP{random.randint(10000, 99999)}",
        "provider_name": provider_name,
        "provider_npi": provider_npi,
        "claim_number": f"CLM{random.randint(10000000, 99999999)}",
        "check_number": f"{random.randint(100000, 999999)}",
        "check_date": rand_date(),
        "service_lines": service_lines,
        "total_billed": total_billed,
        "total_allowed": total_allowed,
        "total_deductible": total_deductible,
        "total_copay": total_copay,
        "total_coinsurance": total_coinsurance,
        "total_plan_paid": total_plan_paid,
        "total_patient_resp": total_patient_resp,
    }


def generate_form(out_pdf: str, data: dict):
    c = canvas.Canvas(out_pdf, pagesize=letter)

    draw_header(c, data)

    y = PAGE_H - 130
    y = draw_member_provider_section(c, y, data)
    y = draw_service_table(c, y, data)
    draw_summary_box(c, y, data)
    y -= 80
    draw_reason_legend(c, y, data)

    c.showPage()
    c.save()


def ground_truth(data: dict) -> dict:
    used_codes = set()
    for line in data["service_lines"]:
        used_codes.update(line["reason_codes"])

    return {
        "form_type": "eob",
        "payer_name": data["payer_name"],
        "member_name": data["member_name"],
        "claim_number": data["claim_number"],
        "check_number": data["check_number"],
        "service_line_count": len(data["service_lines"]),
        "service_line_cpts": [l["cpt"] for l in data["service_lines"]],
        "total_billed": data["total_billed"],
        "total_allowed": data["total_allowed"],
        "total_plan_paid": data["total_plan_paid"],
        "total_patient_resp": data["total_patient_resp"],
        "reason_codes_used": sorted(used_codes),
    }


def generate_dataset(out_dir: str = "dataset/eob", n: int = 25) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"eob": []}
    for i in range(1, n + 1):
        data = generate_scenario()
        pdf_path = out_path / f"eob_{i:03d}.pdf"
        json_path = out_path / f"eob_{i:03d}.json"
        generate_form(str(pdf_path), data)
        json_path.write_text(json.dumps(ground_truth(data), indent=2))
        manifest["eob"].append({"pdf": str(pdf_path), "json": str(json_path)})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    n = int(os.environ.get("N_FORMS", "25"))
    out_dir = os.environ.get("OUT_DIR", "dataset/eob")
    manifest = generate_dataset(out_dir, n)
    print(f"Generated {len(manifest['eob'])} EOB forms in {out_dir}")
