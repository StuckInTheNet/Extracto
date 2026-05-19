"""Generate document types the pipeline has NEVER been trained on.

These are forms with no structuring module — the generic extractor
must handle them. Tests the "no form left behind" principle.

Types generated:
1. Attorney letter of representation
2. Medical records request form
3. Property damage estimate
4. Disability certification / attending physician statement
5. Demand letter summary
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

random.seed(4000)

PAGE_W, PAGE_H = letter
MARGIN = 54


def _draw_kv(c, x, y, label, value, label_w=120):
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + label_w, y, str(value))


def generate_lor(out_pdf: str) -> dict:
    """Letter of Representation — attorney to insurance company."""
    data = {
        "date": "01/15/2025",
        "attorney": "Morgan & Associates, PLLC",
        "attorney_phone": "(312) 555-8901",
        "client": "Jordan Casey Smith",
        "dob": "07/22/1985",
        "doi": "09/14/2024",
        "claim_number": "CLM-2024-89012",
        "adjuster": "Patricia Wells",
        "insurer": "State Farm Insurance",
    }

    c = canvas.Canvas(out_pdf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, PAGE_H - 60, data["attorney"])
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, PAGE_H - 75, "Attorneys at Law")
    c.drawString(MARGIN, PAGE_H - 87, f"Phone: {data['attorney_phone']}")

    c.line(MARGIN, PAGE_H - 95, PAGE_W - MARGIN, PAGE_H - 95)

    y = PAGE_H - 120
    c.setFont("Helvetica", 10)
    c.drawString(MARGIN, y, f"Date: {data['date']}")
    y -= 30
    c.drawString(MARGIN, y, "RE: LETTER OF REPRESENTATION")
    y -= 16
    c.drawString(MARGIN, y, f"Our Client: {data['client']}")
    y -= 14
    c.drawString(MARGIN, y, f"Date of Birth: {data['dob']}")
    y -= 14
    c.drawString(MARGIN, y, f"Date of Injury: {data['doi']}")
    y -= 14
    c.drawString(MARGIN, y, f"Claim Number: {data['claim_number']}")
    y -= 14
    c.drawString(MARGIN, y, f"Adjuster: {data['adjuster']}")
    y -= 30

    body = (
        f"Dear {data['adjuster']},\n\n"
        f"Please be advised that this firm has been retained to represent {data['client']} "
        f"in connection with injuries sustained on {data['doi']}. Please direct all future "
        f"communications regarding this matter to our office.\n\n"
        f"Please forward copies of the following to our office:\n"
        f"1. Copy of the insurance policy\n"
        f"2. Declarations page showing coverage limits\n"
        f"3. Any recorded statements\n"
        f"4. Photographs or surveillance\n\n"
        f"Please do not contact our client directly.\n\n"
        f"Very truly yours,\n\n"
        f"Morgan & Associates, PLLC"
    )
    for line in body.split("\n"):
        c.drawString(MARGIN, y, line)
        y -= 14

    c.showPage()
    c.save()
    return {"form_type": "letter_of_representation", **data}


def generate_property_damage(out_pdf: str) -> dict:
    """Property damage repair estimate."""
    data = {
        "shop": "AutoBody Pro Collision Center",
        "phone": "(555) 234-5678",
        "estimate_date": "10/01/2024",
        "owner": "Jordan Casey Smith",
        "vehicle": "2021 Honda Accord EX",
        "vin": "1HGCV1F34MA012345",
        "mileage": "42,871",
        "insurance": "State Farm Insurance",
        "claim": "CLM-2024-89012",
        "items": [
            ("Front bumper cover R&R", 680.00),
            ("Front bumper reinforcement R&R", 450.00),
            ("Hood - repair", 320.00),
            ("Headlamp assembly LH R&R", 890.00),
            ("Fender LH R&R", 750.00),
            ("Blend adjacent panels", 280.00),
            ("Paint materials", 425.00),
            ("Frame pull - 2 hours", 300.00),
        ],
        "parts_total": 2245.00,
        "labor_total": 1850.00,
        "total": 4095.00,
    }

    c = canvas.Canvas(out_pdf, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, PAGE_H - 50, data["shop"])
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, PAGE_H - 64, f"Phone: {data['phone']}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, PAGE_H - 90, "REPAIR ESTIMATE")
    c.line(MARGIN, PAGE_H - 95, PAGE_W - MARGIN, PAGE_H - 95)

    y = PAGE_H - 115
    _draw_kv(c, MARGIN, y, "Date", data["estimate_date"])
    _draw_kv(c, 320, y, "Claim #", data["claim"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Vehicle Owner", data["owner"])
    _draw_kv(c, 320, y, "Insurance", data["insurance"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Vehicle", data["vehicle"])
    _draw_kv(c, 320, y, "VIN", data["vin"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Mileage", data["mileage"])
    y -= 30

    # Table header
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN, y, "Description")
    c.drawRightString(PAGE_W - MARGIN, y, "Amount")
    c.line(MARGIN, y - 4, PAGE_W - MARGIN, y - 4)
    y -= 18

    c.setFont("Helvetica", 9)
    for desc, amount in data["items"]:
        c.drawString(MARGIN, y, desc)
        c.drawRightString(PAGE_W - MARGIN, y, f"${amount:,.2f}")
        y -= 14

    y -= 8
    c.line(MARGIN, y + 4, PAGE_W - MARGIN, y + 4)
    c.setFont("Helvetica-Bold", 10)
    _draw_kv(c, 350, y, "Parts", f"${data['parts_total']:,.2f}", label_w=80)
    y -= 14
    _draw_kv(c, 350, y, "Labor", f"${data['labor_total']:,.2f}", label_w=80)
    y -= 14
    _draw_kv(c, 350, y, "TOTAL", f"${data['total']:,.2f}", label_w=80)

    c.showPage()
    c.save()
    return {"form_type": "property_damage_estimate", **data}


def generate_disability_cert(out_pdf: str) -> dict:
    """Attending Physician Statement / disability certification."""
    data = {
        "patient": "Jordan Casey Smith",
        "dob": "07/22/1985",
        "ssn_last4": "4728",
        "employer": "Acme Logistics Corp",
        "occupation": "Warehouse Worker",
        "diagnosis": "Lumbar disc herniation (M51.26), Cervical strain (S13.4XXA)",
        "onset_date": "09/14/2024",
        "last_worked": "09/14/2024",
        "return_date": "Undetermined",
        "restrictions": "No lifting > 10 lbs, no prolonged standing, no repetitive bending",
        "prognosis": "Fair",
        "physician": "Sarah Mitchell, MD",
        "npi": "1234567890",
        "signature_date": "01/20/2025",
    }

    c = canvas.Canvas(out_pdf, pagesize=letter)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 50, "ATTENDING PHYSICIAN STATEMENT")
    c.setFont("Helvetica", 8)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 64, "Disability Certification — Confidential Medical Information")
    c.line(MARGIN, PAGE_H - 72, PAGE_W - MARGIN, PAGE_H - 72)

    y = PAGE_H - 95
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "PATIENT INFORMATION")
    y -= 18
    _draw_kv(c, MARGIN, y, "Patient Name", data["patient"])
    _draw_kv(c, 350, y, "SSN (last 4)", data["ssn_last4"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Date of Birth", data["dob"])
    _draw_kv(c, 350, y, "Employer", data["employer"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Occupation", data["occupation"])
    y -= 28

    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "CLINICAL INFORMATION")
    y -= 18
    _draw_kv(c, MARGIN, y, "Primary Diagnosis", data["diagnosis"], label_w=110)
    y -= 16
    _draw_kv(c, MARGIN, y, "Date of Onset", data["onset_date"])
    _draw_kv(c, 350, y, "Last Date Worked", data["last_worked"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Expected Return Date", data["return_date"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Work Restrictions", data["restrictions"], label_w=110)
    y -= 16
    _draw_kv(c, MARGIN, y, "Prognosis", data["prognosis"])
    y -= 40

    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "PHYSICIAN CERTIFICATION")
    y -= 18
    _draw_kv(c, MARGIN, y, "Physician Name", data["physician"])
    _draw_kv(c, 350, y, "NPI", data["npi"])
    y -= 16
    _draw_kv(c, MARGIN, y, "Date", data["signature_date"])
    y -= 24
    c.line(MARGIN, y, MARGIN + 200, y)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, y - 12, "Physician Signature")

    c.showPage()
    c.save()
    return {"form_type": "disability_certification", **data}


def main():
    out_dir = Path("dataset/unknown_forms")
    out_dir.mkdir(parents=True, exist_ok=True)

    forms = [
        ("lor_001.pdf", generate_lor),
        ("property_damage_001.pdf", generate_property_damage),
        ("disability_cert_001.pdf", generate_disability_cert),
    ]

    all_truth = {}
    for filename, generator in forms:
        pdf_path = str(out_dir / filename)
        truth = generator(pdf_path)
        all_truth[filename] = truth
        print(f"Generated: {filename} ({truth['form_type']})")

    (out_dir / "ground_truth.json").write_text(json.dumps(all_truth, indent=2))
    print(f"\nGround truth: {out_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
