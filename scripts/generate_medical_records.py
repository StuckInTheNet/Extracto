"""Generate a synthetic multi-provider medical records bundle.

Simulates what a records retrieval service delivers to a PI attorney:
a single massive PDF containing mixed document types from multiple providers,
with cover sheets, separators, continuation pages, and varied formats.

HIPAA NOTE: All data in these synthetic records is GENERATED. No real
patient information is used. Names, dates, SSNs, conditions, and all
other data are randomized synthetic values.

Document types generated:
- Office visit notes (multi-page narrative with header)
- Radiology/imaging reports
- Lab results
- Physical therapy session notes
- ER reports
- Operative reports
- Discharge summaries
- Billing records (CMS-1500 style)
- Cover sheets / fax headers
- Separator pages between providers
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

random.seed(3000)

PAGE_W, PAGE_H = letter
MARGIN = 54

# --- Synthetic data pools ---

PROVIDERS = [
    {"name": "Sarah Mitchell, MD", "practice": "Mitchell Family Medicine", "npi": "1234567890", "specialty": "Family Medicine"},
    {"name": "James Rodriguez, DO", "practice": "Valley Orthopedics", "npi": "2345678901", "specialty": "Orthopedic Surgery"},
    {"name": "Emily Chen, MD", "practice": "Central Neurology Associates", "npi": "3456789012", "specialty": "Neurology"},
    {"name": "Michael Thompson, DC", "practice": "Spine & Sport Chiropractic", "npi": "4567890123", "specialty": "Chiropractic"},
    {"name": "Priya Patel, MD", "practice": "Patel Pain Management", "npi": "5678901234", "specialty": "Pain Management"},
    {"name": "David Kim, MD", "practice": "Valley Medical Center ER", "npi": "6789012345", "specialty": "Emergency Medicine"},
    {"name": "Lisa Washington, PT", "practice": "Central Physical Therapy", "npi": "7890123456", "specialty": "Physical Therapy"},
    {"name": "Robert Garcia, MD", "practice": "Imaging Associates", "npi": "8901234567", "specialty": "Radiology"},
]

PATIENT = {
    "name": "Jordan Casey Smith",
    "dob": "07/22/1985",
    "mrn": "MRN-482901",
    "ssn_last4": "4728",
    "address": "2376 Elm Street, Springfield, IL 62704",
    "phone": "(217) 555-0142",
    "insurance": "Blue Cross Blue Shield",
    "policy": "BCB8291047",
}

DIAGNOSES = [
    "Cervical strain (S13.4XXA)",
    "Lumbar disc herniation (M51.26)",
    "Post-traumatic headache (G44.311)",
    "Cervicalgia (M54.2)",
    "Low back pain (M54.5)",
    "Right shoulder pain (M25.511)",
    "Anxiety disorder (F41.1)",
    "Muscle spasm of back (M62.830)",
]

MEDICATIONS = [
    "Ibuprofen 800mg TID",
    "Cyclobenzaprine 10mg TID",
    "Gabapentin 300mg TID",
    "Meloxicam 15mg daily",
    "Tramadol 50mg PRN",
    "Methocarbamol 750mg QID",
]

VISIT_TEMPLATES = {
    "office_visit": {
        "sections": ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS",
                      "PHYSICAL EXAMINATION", "ASSESSMENT", "PLAN"],
    },
    "er_report": {
        "sections": ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "EMERGENCY DEPARTMENT COURSE",
                      "PHYSICAL EXAM", "DIAGNOSTIC STUDIES", "DIAGNOSIS", "DISPOSITION"],
    },
    "radiology": {
        "sections": ["EXAM", "CLINICAL INDICATION", "TECHNIQUE", "FINDINGS", "IMPRESSION"],
    },
    "pt_note": {
        "sections": ["SUBJECTIVE", "OBJECTIVE", "TREATMENT PROVIDED", "RESPONSE TO TREATMENT", "PLAN"],
    },
    "operative": {
        "sections": ["PREOPERATIVE DIAGNOSIS", "POSTOPERATIVE DIAGNOSIS", "PROCEDURE",
                      "ANESTHESIA", "FINDINGS", "DESCRIPTION OF PROCEDURE", "COMPLICATIONS",
                      "ESTIMATED BLOOD LOSS", "DISPOSITION"],
    },
    "discharge": {
        "sections": ["ADMISSION DATE", "DISCHARGE DATE", "ADMITTING DIAGNOSIS",
                      "DISCHARGE DIAGNOSIS", "HOSPITAL COURSE", "DISCHARGE MEDICATIONS",
                      "FOLLOW-UP INSTRUCTIONS", "ACTIVITY RESTRICTIONS"],
    },
    "lab_result": {
        "sections": ["TEST", "SPECIMEN", "RESULTS", "REFERENCE RANGE", "INTERPRETATION"],
    },
}

LOREM = (
    "Patient reports continued pain in the cervical and lumbar regions. "
    "Symptoms have been persistent since the date of injury. Pain is described "
    "as sharp and radiating, worse with activity and prolonged sitting. "
    "Patient has been compliant with prescribed medications and physical therapy. "
    "Range of motion remains limited. Neurological examination is within normal "
    "limits. No new focal deficits noted. Will continue current treatment plan "
    "and re-evaluate in four weeks. Patient is to avoid heavy lifting and "
    "repetitive bending. Work restrictions remain in place."
)


def rand_date_between(start: str, end: str) -> str:
    s = datetime.strptime(start, "%m/%d/%Y")
    e = datetime.strptime(end, "%m/%d/%Y")
    d = s + timedelta(days=random.randint(0, (e - s).days))
    return d.strftime("%m/%d/%Y")


def _fill_section_text() -> str:
    """Generate a paragraph of synthetic clinical text."""
    sentences = LOREM.split(". ")
    n = random.randint(2, 6)
    selected = random.sample(sentences, min(n, len(sentences)))
    return ". ".join(selected) + "."


# --- Page rendering ---

def draw_provider_header(c: canvas.Canvas, provider: dict, doc_type: str, dos: str, page_num: int | None = None):
    """Draw a provider letterhead header at the top of a page."""
    # Practice name (large, bold)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, PAGE_H - 50, provider["practice"])

    # Provider details
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, PAGE_H - 65, f"{provider['name']} | {provider['specialty']} | NPI: {provider['npi']}")

    # Horizontal line
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(MARGIN, PAGE_H - 72, PAGE_W - MARGIN, PAGE_H - 72)

    # Document type + date
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, PAGE_H - 88, doc_type.upper())
    c.setFont("Helvetica", 10)
    c.drawString(PAGE_W - MARGIN - 120, PAGE_H - 88, f"Date: {dos}")

    # Patient info line
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, PAGE_H - 102, f"Patient: {PATIENT['name']}   DOB: {PATIENT['dob']}   MRN: {PATIENT['mrn']}")

    if page_num is not None:
        c.setFont("Helvetica", 8)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 102, f"Page {page_num}")

    return PAGE_H - 115


def draw_section(c: canvas.Canvas, y: float, section_name: str, content: str) -> float:
    """Draw a labeled section with wrapped text."""
    if y < 100:
        return y  # no room

    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, f"{section_name}:")
    y -= 14

    c.setFont("Helvetica", 9)
    words = content.split()
    line = ""
    max_w = PAGE_W - 2 * MARGIN
    for word in words:
        test = line + (" " if line else "") + word
        if c.stringWidth(test, "Helvetica", 9) <= max_w:
            line = test
        else:
            c.drawString(MARGIN, y, line)
            y -= 12
            line = word
            if y < 80:
                break
    if line and y >= 80:
        c.drawString(MARGIN, y, line)
        y -= 12

    return y - 6


def draw_cover_sheet(c: canvas.Canvas, provider: dict, records_service: str):
    """Draw a records retrieval cover sheet."""
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 80, records_service)

    c.setFont("Helvetica", 11)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 110, "MEDICAL RECORDS TRANSMITTAL")

    c.setStrokeColor(colors.black)
    c.line(MARGIN, PAGE_H - 120, PAGE_W - MARGIN, PAGE_H - 120)

    y = PAGE_H - 150
    c.setFont("Helvetica", 10)
    fields = [
        ("Patient", PATIENT["name"]),
        ("Date of Birth", PATIENT["dob"]),
        ("Records From", provider["practice"]),
        ("Provider", provider["name"]),
        ("Date Prepared", rand_date_between("01/01/2025", "03/01/2025")),
    ]
    for label, value in fields:
        c.drawString(MARGIN, y, f"{label}:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(MARGIN + 120, y, value)
        c.setFont("Helvetica", 10)
        y -= 22

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(MARGIN, y - 20, "CONFIDENTIAL — Protected Health Information enclosed.")
    c.drawString(MARGIN, y - 34, "Unauthorized disclosure is prohibited under HIPAA (45 CFR Parts 160, 164).")


def draw_separator(c: canvas.Canvas, label: str):
    """Draw a section separator page."""
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(PAGE_W / 2, PAGE_H / 2 + 20, "=" * 30)
    c.drawCentredString(PAGE_W / 2, PAGE_H / 2, label)
    c.drawCentredString(PAGE_W / 2, PAGE_H / 2 - 20, "=" * 30)


def render_document(c: canvas.Canvas, provider: dict, doc_type: str, dos: str, n_pages: int = 1) -> int:
    """Render a multi-page clinical document. Returns number of pages rendered."""
    template = VISIT_TEMPLATES.get(doc_type, VISIT_TEMPLATES["office_visit"])
    sections = template["sections"]

    # Assign diagnoses and medications for this visit
    visit_dx = random.sample(DIAGNOSES, k=random.randint(1, 3))
    visit_meds = random.sample(MEDICATIONS, k=random.randint(1, 3))

    pages_rendered = 0
    section_idx = 0

    for page_num in range(1, n_pages + 1):
        y = draw_provider_header(c, provider, doc_type, dos, page_num=page_num if n_pages > 1 else None)
        pages_rendered += 1

        while section_idx < len(sections) and y > 120:
            section = sections[section_idx]

            # Generate content based on section name
            if "DIAGNOSIS" in section or "ASSESSMENT" in section:
                content = "; ".join(visit_dx)
            elif "MEDICATION" in section or "PLAN" in section:
                content = "; ".join(visit_meds) + ". " + _fill_section_text()
            elif "BLOOD LOSS" in section:
                content = f"Estimated {random.choice([50, 100, 150, 200, 250])} mL"
            elif "COMPLICATIONS" in section:
                content = random.choice(["None", "None", "Minor bleeding, controlled"])
            elif "ANESTHESIA" in section:
                content = random.choice(["General endotracheal", "Conscious sedation", "Local with MAC"])
            else:
                content = _fill_section_text()

            y = draw_section(c, y, section, content)
            section_idx += 1

        # Signature at bottom of last page
        if page_num == n_pages or section_idx >= len(sections):
            if y > 80:
                c.setFont("Helvetica", 9)
                c.line(MARGIN, 70, MARGIN + 200, 70)
                c.drawString(MARGIN, 58, f"{provider['name']}  —  {dos}")

        if page_num < n_pages:
            c.showPage()

    return pages_rendered


# --- Bundle generation ---

def generate_encounter(provider: dict, dos: str, doc_type: str) -> dict:
    """Define a single clinical encounter."""
    if doc_type == "pt_note":
        n_pages = 1
    elif doc_type in ("operative", "discharge", "er_report"):
        n_pages = random.randint(2, 4)
    elif doc_type == "lab_result":
        n_pages = 1
    elif doc_type == "radiology":
        n_pages = random.randint(1, 2)
    else:
        n_pages = random.randint(1, 3)

    return {
        "provider": provider,
        "dos": dos,
        "doc_type": doc_type,
        "n_pages": n_pages,
    }


def generate_records_bundle(out_pdf: str, target_pages: int = 200) -> dict:
    """Generate a realistic multi-provider medical records bundle.

    Returns ground truth with document boundaries, providers, and dates.
    """
    c = canvas.Canvas(out_pdf, pagesize=letter)
    records_service = random.choice([
        "National Records Retrieval Inc.",
        "MedRecords Express LLC",
        "ChartSwap Medical Records",
    ])

    # Plan the encounters: each provider has multiple visits over time
    injury_date = "09/14/2024"
    encounters: list[dict] = []

    # Primary care (most visits)
    pcp = PROVIDERS[0]
    for i in range(8):
        dos = rand_date_between("09/14/2024", "06/01/2025")
        encounters.append(generate_encounter(pcp, dos, "office_visit"))

    # Orthopedics
    ortho = PROVIDERS[1]
    for i in range(4):
        dos = rand_date_between("10/01/2024", "04/01/2025")
        encounters.append(generate_encounter(ortho, dos, "office_visit"))
    encounters.append(generate_encounter(ortho, "02/15/2025", "operative"))

    # Neurology
    neuro = PROVIDERS[2]
    for i in range(3):
        dos = rand_date_between("10/15/2024", "03/15/2025")
        encounters.append(generate_encounter(neuro, dos, "office_visit"))

    # Chiropractic (many short visits)
    chiro = PROVIDERS[3]
    for i in range(15):
        dos = rand_date_between("10/01/2024", "06/01/2025")
        encounters.append(generate_encounter(chiro, dos, "pt_note"))

    # Pain management
    pain = PROVIDERS[4]
    for i in range(3):
        dos = rand_date_between("11/01/2024", "05/01/2025")
        encounters.append(generate_encounter(pain, dos, "office_visit"))

    # ER visits
    er = PROVIDERS[5]
    encounters.append(generate_encounter(er, injury_date, "er_report"))
    encounters.append(generate_encounter(er, "12/20/2024", "er_report"))

    # PT sessions
    pt = PROVIDERS[6]
    for i in range(12):
        dos = rand_date_between("10/15/2024", "05/15/2025")
        encounters.append(generate_encounter(pt, dos, "pt_note"))

    # Radiology
    rad = PROVIDERS[7]
    for i in range(4):
        dos = rand_date_between("09/20/2024", "03/01/2025")
        encounters.append(generate_encounter(rad, dos, "radiology"))

    # Lab results (sprinkled in)
    for i in range(3):
        lab_provider = random.choice([pcp, ortho, neuro])
        dos = rand_date_between("10/01/2024", "05/01/2025")
        encounters.append(generate_encounter(
            {"name": "Quest Diagnostics", "practice": "Quest Diagnostics",
             "npi": "9012345678", "specialty": "Laboratory"},
            dos, "lab_result"
        ))

    # Sort by provider, then date
    encounters.sort(key=lambda e: (e["provider"]["practice"], e["dos"]))

    # Group by provider for cover sheets
    by_provider: dict[str, list[dict]] = {}
    for enc in encounters:
        key = enc["provider"]["practice"]
        by_provider.setdefault(key, []).append(enc)

    # Render the bundle
    ground_truth: list[dict] = []
    current_page = 1

    for provider_name, provider_encounters in by_provider.items():
        provider = provider_encounters[0]["provider"]

        # Cover sheet for this provider's records
        draw_cover_sheet(c, provider, records_service)
        ground_truth.append({
            "start_page": current_page,
            "end_page": current_page,
            "provider": provider_name,
            "doc_type": "cover_sheet",
            "dos": None,
        })
        c.showPage()
        current_page += 1

        # Separator
        draw_separator(c, f"Records from {provider_name}")
        ground_truth.append({
            "start_page": current_page,
            "end_page": current_page,
            "provider": provider_name,
            "doc_type": "separator",
            "dos": None,
        })
        c.showPage()
        current_page += 1

        # Render each encounter for this provider
        for enc in provider_encounters:
            start = current_page
            pages = render_document(c, enc["provider"], enc["doc_type"], enc["dos"], enc["n_pages"])
            ground_truth.append({
                "start_page": start,
                "end_page": start + pages - 1,
                "provider": provider_name,
                "doc_type": enc["doc_type"],
                "dos": enc["dos"],
            })
            current_page += pages
            c.showPage()
            current_page += 1  # showPage adds a page break

        # Adjust: showPage after last encounter creates an extra blank
        current_page -= 1

    c.save()

    # Remove trailing page count adjustment
    total_pages = current_page - 1

    manifest = {
        "source_pdf": out_pdf,
        "total_pages": total_pages,
        "patient": PATIENT,
        "injury_date": injury_date,
        "records_service": records_service,
        "provider_count": len(by_provider),
        "encounter_count": len([g for g in ground_truth if g["doc_type"] not in ("cover_sheet", "separator")]),
        "documents": ground_truth,
    }

    return manifest


def main():
    out_dir = Path("dataset/medical_records")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = str(out_dir / "records_bundle.pdf")
    manifest = generate_records_bundle(pdf_path, target_pages=200)

    gt_path = out_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(manifest, indent=2))

    # Print summary (NO PHI — just counts and providers)
    print(f"Generated: {pdf_path}")
    print(f"Total pages: {manifest['total_pages']}")
    print(f"Providers: {manifest['provider_count']}")
    print(f"Encounters: {manifest['encounter_count']}")
    print(f"Ground truth: {gt_path}")
    print()
    print("Provider breakdown:")
    from collections import Counter
    by_prov = Counter(d["provider"] for d in manifest["documents"] if d["doc_type"] not in ("cover_sheet", "separator"))
    for prov, count in sorted(by_prov.items()):
        print(f"  {prov}: {count} encounters")


if __name__ == "__main__":
    main()
