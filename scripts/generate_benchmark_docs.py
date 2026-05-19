import os
import json
import random
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron",
    "Quinn", "Rowan", "Skyler", "Peyton", "Reese", "Sawyer", "Logan", "Hayden", "Emerson", "Finley"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson",
    "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White"
]
STREETS = [
    "Oak St", "Maple Ave", "Pine Rd", "Cedar Ln", "Elm St", "Birch Blvd", "Willow Way", "Hickory Dr", "Ash Ct", "Spruce Ter"
]
CITIES = [
    "Springfield", "Riverton", "Greenville", "Fairview", "Franklin", "Georgetown", "Clinton", "Madison", "Salem", "Arlington"
]
STATES = ["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]

ALLERGY_OPTIONS = ["Penicillin", "Peanuts", "Latex", "Shellfish", "Pollen", "Dust", "None Known"]
SYMPTOMS = ["Fever", "Cough", "Headache", "Fatigue", "Shortness of breath", "Nausea", "Dizziness", "Chest pain"]
INSURANCE_TYPES = ["HMO", "PPO", "EPO", "POS"]
MED_TITLES = ["Patient Intake Form", "New Patient Information", "Medical Intake"]
INS_TITLES = ["Insurance Claim Form", "Health Insurance Claim", "Claim Submission"]

random.seed(42)


def rand_phone():
    return f"({random.randint(200,989)}) {random.randint(200,989)}-{random.randint(1000,9999)}"


def rand_date(start_year=1940, end_year=2005):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta_days = (end - start).days
    d = start + timedelta(days=random.randint(0, delta_days))
    return d.strftime("%m/%d/%Y")


def rand_insurance_id():
    return f"{random.choice(['A','B','C','D','E','F'])}{random.randint(1000000,9999999)}"


def rand_address():
    num = random.randint(100, 9999)
    street = random.choice(STREETS)
    city = random.choice(CITIES)
    state = random.choice(STATES)
    zipc = random.randint(10000, 99999)
    return f"{num} {street}", f"{city}, {state} {zipc}"


def draw_checkbox(c: canvas.Canvas, x, y, size=12, state="empty", style=None, line_w=1, rounded=False):
    # state: empty, filled, x, check
    c.setStrokeColor(colors.black)
    c.setLineWidth(line_w)
    if rounded:
        c.roundRect(x, y, size, size, radius=size*0.2, stroke=1, fill=0)
    else:
        c.rect(x, y, size, size, stroke=1, fill=0)
    if state == "filled":
        c.setFillColor(colors.black)
        c.rect(x+2, y+2, size-4, size-4, stroke=0, fill=1)
        c.setFillColor(colors.black)
    elif state == "x":
        c.line(x+2, y+2, x+size-2, y+size-2)
        c.line(x+2, y+size-2, x+size-2, y+2)
    elif state == "check":
        # simple checkmark
        c.setLineWidth(max(2, line_w))
        c.line(x+2, y+size/2-1, x+size/2-1, y+2)
        c.line(x+size/2-1, y+2, x+size-2, y+size-2)
        c.setLineWidth(line_w)
    return (x, y, size, size)


def draw_radio(c: canvas.Canvas, x, y, radius=6, selected=False, line_w=1):
    c.setStrokeColor(colors.black)
    c.setLineWidth(line_w)
    c.circle(x+radius, y+radius, radius, stroke=1, fill=0)
    if selected:
        c.setFillColor(colors.black)
        c.circle(x+radius, y+radius, radius-3, stroke=0, fill=1)
        c.setFillColor(colors.black)
    return (x, y, radius*2, radius*2)


def draw_two_column_guides(c: canvas.Canvas, page_w, page_h, margin, gutter):
    # Optional visual gutter line (not printed by default)
    pass


def header(c: canvas.Canvas, title, page_w, page_h, margin):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - margin + 4, title)
    # Fake logo box
    c.rect(page_w - margin - 120, page_h - margin - 8, 120, 22, stroke=1, fill=0)
    c.setFont("Helvetica", 8)
    c.drawRightString(page_w - margin - 6, page_h - margin - 2, "ACME Health Group")


def draw_kv(c: canvas.Canvas, x, y, label, value, width=250):
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"{label}:")
    c.setFont("Helvetica", 11)
    c.drawString(x + 120, y, value)
    # underline
    c.setStrokeColor(colors.lightgrey)
    c.line(x + 120, y - 2, x + width, y - 2)
    c.setStrokeColor(colors.black)


def draw_group_checkboxes(c: canvas.Canvas, x, y, label, options, selected_indices=None, column_gap=140, row_gap=18, controls=None, size=12, line_w=1, jitter=0, rounded=False):
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, label)
    y -= 6
    c.setFont("Helvetica", 10)
    if selected_indices is None:
        selected_indices = set()
    start_y = y
    col = 0
    for i, opt in enumerate(options):
        cx = x + col * column_gap
        cy = start_y - (i % 6) * row_gap
        state = "empty"
        if i in selected_indices:
            state = random.choice(["filled", "x", "check"])  # varied marks
        jx = cx + random.randint(-jitter, jitter)
        jy = cy - 10 + random.randint(-jitter, jitter)
        bx = draw_checkbox(c, jx, jy, size=size, state=state, line_w=line_w, rounded=rounded)
        c.drawString(jx + size + 6, jy + 4, opt)
        if controls is not None:
            controls.append({"kind":"checkbox","label":opt,"selected": (i in selected_indices), "bbox_pt": [bx[0], bx[1], bx[2], bx[3]]})
        if (i + 1) % 6 == 0:
            col += 1
    return start_y - ((len(options)-1) % 6) * row_gap - 36


def draw_yes_no(c: canvas.Canvas, x, y, label, value_yes=False, as_radio=False, controls=None, size=12, line_w=1, jitter=0, rounded=False):
    c.setFont("Helvetica", 10)
    c.drawString(x, y, label)
    y -= 2
    if as_radio:
        b1 = draw_radio(c, x + 120 + random.randint(-jitter, jitter), y - 10 + random.randint(-jitter, jitter), selected=value_yes, line_w=line_w, radius=max(5, size//2))
        c.drawString(b1[0] + b1[2] + 6, b1[1] + b1[3] - 6, "Yes")
        b2 = draw_radio(c, x + 200 + random.randint(-jitter, jitter), y - 10 + random.randint(-jitter, jitter), selected=not value_yes, line_w=line_w, radius=max(5, size//2))
        c.drawString(b2[0] + b2[2] + 6, b2[1] + b2[3] - 6, "No")
        if controls is not None:
            controls.append({"kind":"radio","label":"Yes","selected": bool(value_yes), "bbox_pt": [b1[0], b1[1], b1[2], b1[3]], "parent": label})
            controls.append({"kind":"radio","label":"No","selected": bool(not value_yes), "bbox_pt": [b2[0], b2[1], b2[2], b2[3]], "parent": label})
    else:
        b1 = draw_checkbox(c, x + 118 + random.randint(-jitter, jitter), y - 10 + random.randint(-jitter, jitter), state="check" if value_yes else "empty", size=size, line_w=line_w, rounded=rounded)
        c.drawString(b1[0] + b1[2] + 6, b1[1] + b1[3] - 6, "Yes")
        b2 = draw_checkbox(c, x + 198 + random.randint(-jitter, jitter), y - 10 + random.randint(-jitter, jitter), state="check" if not value_yes else "empty", size=size, line_w=line_w, rounded=rounded)
        c.drawString(b2[0] + b2[2] + 6, b2[1] + b2[3] - 6, "No")
        if controls is not None:
            controls.append({"kind":"checkbox","label":"Yes","selected": bool(value_yes), "bbox_pt": [b1[0], b1[1], b1[2], b1[3]], "parent": label})
            controls.append({"kind":"checkbox","label":"No","selected": bool(not value_yes), "bbox_pt": [b2[0], b2[1], b2[2], b2[3]], "parent": label})


def medical_form(c: canvas.Canvas, person, meta):
    page_w, page_h = letter
    margin = 54
    gutter = 24
    col_w = (page_w - 2 * margin - gutter) / 2
    left_x = margin
    right_x = margin + col_w + gutter
    y = page_h - margin - 24

    header(c, random.choice(MED_TITLES), page_w, page_h, margin)
    controls: List[dict] = []

    # Left column
    draw_kv(c, left_x, y, "Patient Name", person["name"]) ; y -= 20
    draw_kv(c, left_x, y, "Date of Birth", person["dob"]) ; y -= 20
    draw_kv(c, left_x, y, "Phone", person["phone"]) ; y -= 20
    addr1, addr2 = person["address"]
    draw_kv(c, left_x, y, "Address", addr1) ; y -= 20
    draw_kv(c, left_x, y, "", addr2) ; y -= 26
    sex = person["sex"]
    c.setFont("Helvetica", 10)
    c.drawString(left_x, y, "Sex:")
    sz = random.randint(10, 16)
    lw = random.choice([1, 1, 2])
    jitter = random.randint(0, 3)
    b_m = draw_radio(c, left_x + 40, y - 10, selected=(sex == "Male"), line_w=lw, radius=max(5, sz//2))
    c.drawString(left_x + 60, y - 6, "Male")
    b_f = draw_radio(c, left_x + 120, y - 10, selected=(sex == "Female"), line_w=lw, radius=max(5, sz//2))
    c.drawString(left_x + 140, y - 6, "Female")
    b_o = draw_radio(c, left_x + 200, y - 10, selected=(sex == "Other"), line_w=lw, radius=max(5, sz//2))
    c.drawString(left_x + 220, y - 6, "Other")
    controls += [
        {"kind":"radio","label":"Male","selected": sex=="Male","bbox_pt":[b_m[0],b_m[1],b_m[2],b_m[3]],"parent":"Sex"},
        {"kind":"radio","label":"Female","selected": sex=="Female","bbox_pt":[b_f[0],b_f[1],b_f[2],b_f[3]],"parent":"Sex"},
        {"kind":"radio","label":"Other","selected": sex=="Other","bbox_pt":[b_o[0],b_o[1],b_o[2],b_o[3]],"parent":"Sex"},
    ]
    y -= 28
    draw_yes_no(c, left_x, y, "Smoker", meta["smoker"], as_radio=True, controls=controls, size=sz, line_w=lw, jitter=jitter)
    y -= 28
    draw_yes_no(c, left_x, y, "Diabetic", meta["diabetic"], as_radio=False, controls=controls, size=sz, line_w=lw, jitter=jitter, rounded=bool(random.getrandbits(1)))
    y -= 34
    # Allergies group (left)
    selected = set(i for i, a in enumerate(ALLERGY_OPTIONS) if a in meta["allergies"])
    y = draw_group_checkboxes(c, left_x, y, "Allergies (check all that apply)", ALLERGY_OPTIONS, selected, controls=controls, size=sz, line_w=lw, jitter=jitter, rounded=bool(random.getrandbits(1)))

    # Right column
    ry = page_h - margin - 24
    draw_kv(c, right_x, ry, "MRN", meta["mrn"]) ; ry -= 20
    draw_kv(c, right_x, ry, "Primary Physician", meta["physician"]) ; ry -= 20
    draw_kv(c, right_x, ry, "Emergency Contact", meta["emergency_contact"]) ; ry -= 20
    draw_kv(c, right_x, ry, "Contact Phone", meta["emergency_phone"]) ; ry -= 30
    ry = draw_group_checkboxes(c, right_x, ry, "Current Symptoms", SYMPTOMS, set(meta["symptoms"]), controls=controls, size=sz, line_w=lw, jitter=jitter, rounded=bool(random.getrandbits(1)))
    ry -= 6
    c.setFont("Helvetica", 10)
    c.drawString(right_x, ry, "Signature: ________________________________   Date: ___________")
    return controls

def insurance_form(c: canvas.Canvas, person, meta):
    page_w, page_h = letter
    margin = 54
    gutter = 24
    col_w = (page_w - 2 * margin - gutter) / 2
    left_x = margin
    right_x = margin + col_w + gutter
    y = page_h - margin - 24

    header(c, random.choice(INS_TITLES), page_w, page_h, margin)
    controls: List[dict] = []

    # Left column
    draw_kv(c, left_x, y, "Policy Holder", person["name"]) ; y -= 20
    draw_kv(c, left_x, y, "DOB", person["dob"]) ; y -= 20
    draw_kv(c, left_x, y, "Policy #", meta["policy_id"]) ; y -= 20
    draw_kv(c, left_x, y, "Insurance Type", meta["ins_type"]) ; y -= 26
    sz = random.randint(10, 16)
    lw = random.choice([1, 1, 2])
    jitter = random.randint(0, 3)
    draw_yes_no(c, left_x, y, "Is this work-related?", meta["work_related"], as_radio=True, controls=controls, size=sz, line_w=lw, jitter=jitter) ; y -= 28
    draw_yes_no(c, left_x, y, "Auto accident?", meta["auto_accident"], as_radio=False, controls=controls, size=sz, line_w=lw, jitter=jitter, rounded=bool(random.getrandbits(1))) ; y -= 28

    # Right column
    ry = page_h - margin - 24
    addr1, addr2 = person["address"]
    draw_kv(c, right_x, ry, "Address", addr1) ; ry -= 20
    draw_kv(c, right_x, ry, "", addr2) ; ry -= 20
    draw_kv(c, right_x, ry, "Phone", person["phone"]) ; ry -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(right_x, ry, "Claim Type") ; ry -= 6
    claim_opts = ["Visit", "Procedure", "Medication", "Other"]
    sel = meta["claim_type"]
    for i, opt in enumerate(claim_opts):
        b = draw_radio(c, right_x + 2, ry - 10, selected=(opt == sel), line_w=lw, radius=max(5, sz//2))
        c.setFont("Helvetica", 10)
        c.drawString(right_x + 22, ry - 6, opt)
        controls.append({"kind":"radio","label":opt,"selected": bool(opt==sel), "bbox_pt":[b[0],b[1],b[2],b[3]], "parent":"Claim Type"})
        ry -= 18

    # Services checkboxes
    ry -= 6
    services = ["X-Ray", "MRI", "Lab Work", "Physical Therapy", "Consultation", "Surgery"]
    selected = set(random.sample(range(len(services)), k=random.randint(1, 3)))
    draw_group_checkboxes(c, right_x, ry, "Services Rendered", services, selected, controls=controls, size=sz, line_w=lw, jitter=jitter, rounded=bool(random.getrandbits(1)))
    return controls


def medical_form_v2(c: canvas.Canvas, person, meta):
    page_w, page_h = letter
    margin = 54
    gutter = 18
    col_w = (page_w - 2 * margin - 2 * gutter) / 3
    xs = [margin + i * (col_w + gutter) for i in range(3)]
    header(c, random.choice(MED_TITLES), page_w, page_h, margin)
    y = page_h - margin - 24
    draw_kv(c, xs[0], y, "Patient Name", person["name"]) ; y -= 20
    draw_kv(c, xs[0], y, "DOB", person["dob"]) ; y -= 20
    draw_kv(c, xs[0], y, "Phone", person["phone"]) ; y -= 26
    sex = person["sex"]
    c.setFont("Helvetica", 10)
    c.drawString(xs[0], y, "Gender:")
    draw_radio(c, xs[0] + 60, y - 10, selected=(sex == "Male")) ; c.drawString(xs[0] + 80, y - 6, "Male")
    draw_radio(c, xs[0] + 130, y - 10, selected=(sex == "Female")) ; c.drawString(xs[0] + 150, y - 6, "Female")
    y -= 28
    draw_yes_no(c, xs[0], y, "Smoker", meta["smoker"], as_radio=False) ; y -= 26
    draw_yes_no(c, xs[0], y, "Diabetic", meta["diabetic"], as_radio=True) ; y -= 26
    # Allergies in middle
    selected = set(i for i,a in enumerate(ALLERGY_OPTIONS) if a in meta["allergies"])
    y2 = page_h - margin - 24
    draw_group_checkboxes(c, xs[1], y2, "Allergies", ALLERGY_OPTIONS, selected)
    # Symptoms on right
    y3 = page_h - margin - 24
    draw_group_checkboxes(c, xs[2], y3, "Symptoms", SYMPTOMS, set(meta["symptoms"]))
    return []


def insurance_form_v2(c: canvas.Canvas, person, meta):
    page_w, page_h = letter
    margin = 54
    header(c, random.choice(INS_TITLES), page_w, page_h, margin)
    left_x = margin
    y = page_h - margin - 24
    draw_kv(c, left_x, y, "Policy Holder", person["name"]) ; y -= 20
    draw_kv(c, left_x, y, "Member ID", meta["policy_id"]) ; y -= 20
    draw_kv(c, left_x, y, "Type", meta["ins_type"]) ; y -= 26
    draw_yes_no(c, left_x, y, "Work-related?", meta["work_related"], as_radio=False) ; y -= 26
    draw_yes_no(c, left_x, y, "Auto accident?", meta["auto_accident"], as_radio=True) ; y -= 26
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x, y, "Claim Type") ; y -= 6
    for opt in ["Visit","Procedure","Medication","Other"]:
        draw_radio(c, left_x+2, y-10, selected=(opt==meta["claim_type"]))
        c.setFont("Helvetica",10); c.drawString(left_x+22, y-6, opt)
        y -= 18
    return []


def hipaa_auth_form(c: canvas.Canvas, person, meta):
    page_w, page_h = letter
    margin = 54
    header(c, "HIPAA Authorization", page_w, page_h, margin)
    y = page_h - margin - 24
    draw_kv(c, margin, y, "Patient", person["name"]) ; y -= 20
    draw_kv(c, margin, y, "DOB", person["dob"]) ; y -= 26
    draw_yes_no(c, margin, y, "Authorize release", value_yes=True, as_radio=True) ; y -= 26
    purposes = ["Treatment","Payment","Operations","Legal","Other"]
    sel = set(random.sample(range(len(purposes)), k=random.randint(1,3)))
    draw_group_checkboxes(c, margin, y, "Purpose of disclosure", purposes, sel) ; y -= 120
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "Expiration Date: ")
    c.drawString(margin+120, y, rand_date(2024,2026))
    return []


def random_person():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    dob = rand_date()
    phone = rand_phone()
    address = rand_address()
    sex = random.choice(["Male", "Female", "Other"])
    return {"name": name, "dob": dob, "phone": phone, "address": address, "sex": sex}


def random_medical_meta():
    mrn = f"{random.randint(100000,999999)}-{random.randint(10,99)}"
    physician = f"Dr. {random.choice(LAST_NAMES)}"
    emergency_contact = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    emergency_phone = rand_phone()
    smoker = random.choice([True, False])
    diabetic = random.choice([True, False])
    allergies = random.sample(ALLERGY_OPTIONS[:-1], k=random.randint(0, 3))
    if not allergies:
        allergies = ["None Known"]
    symptoms_idx = sorted(random.sample(range(len(SYMPTOMS)), k=random.randint(1, 4)))
    return {
        "mrn": mrn,
        "physician": physician,
        "emergency_contact": emergency_contact,
        "emergency_phone": emergency_phone,
        "smoker": smoker,
        "diabetic": diabetic,
        "allergies": allergies,
        "symptoms": symptoms_idx,
    }


def random_insurance_meta():
    policy_id = rand_insurance_id()
    ins_type = random.choice(INSURANCE_TYPES)
    work_related = random.choice([True, False])
    auto_accident = random.choice([True, False])
    claim_type = random.choice(["Visit", "Procedure", "Medication", "Other"])
    return {
        "policy_id": policy_id,
        "ins_type": ins_type,
        "work_related": work_related,
        "auto_accident": auto_accident,
        "claim_type": claim_type,
    }


def ensure_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, "medical"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "insurance"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "auth"), exist_ok=True)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_dataset(base_dir="dataset", n_medical=13, n_insurance=12, n_auth=0):
    ensure_dirs(base_dir)
    manifest = {"medical": [], "insurance": []}

    # Medical forms
    for i in range(1, n_medical + 1):
        person = random_person()
        meta = random_medical_meta()
        pdf_name = f"medical_{i:03d}.pdf"
        json_name = f"medical_{i:03d}.json"
        pdf_path = os.path.join(base_dir, "medical", pdf_name)
        json_path = os.path.join(base_dir, "medical", json_name)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        if random.random()<0.5:
            ctrls = medical_form(c, person, meta)
        else:
            ctrls = medical_form_v2(c, person, meta)
        c.showPage()
        c.save()
        truth = {"person": person, "meta": meta, "type": "medical", "controls": ctrls}
        save_json(json_path, truth)
        manifest["medical"].append({"pdf": pdf_path, "json": json_path})

    # Insurance forms
    for i in range(1, n_insurance + 1):
        person = random_person()
        meta = random_insurance_meta()
        pdf_name = f"insurance_{i:03d}.pdf"
        json_name = f"insurance_{i:03d}.json"
        pdf_path = os.path.join(base_dir, "insurance", pdf_name)
        json_path = os.path.join(base_dir, "insurance", json_name)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        if random.random()<0.5:
            ctrls = insurance_form(c, person, meta)
        else:
            ctrls = insurance_form_v2(c, person, meta)
        c.showPage()
        c.save()
        truth = {"person": person, "meta": meta, "type": "insurance", "controls": ctrls}
        save_json(json_path, truth)
        manifest["insurance"].append({"pdf": pdf_path, "json": json_path})

    # Auth forms for splitting realism
    for i in range(1, n_auth+1):
        person = random_person()
        meta = random_medical_meta()
        pdf_name = f"auth_{i:03d}.pdf"
        pdf_path = os.path.join(base_dir, "auth", pdf_name)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        hipaa_auth_form(c, person, meta)
        c.showPage(); c.save()
    save_json(os.path.join(base_dir, "manifest.json"), manifest)
    return manifest


if __name__ == "__main__":
    out_dir = os.environ.get("OUT_DIR", "dataset")
    n_med = int(os.environ.get("N_MEDICAL", "13"))
    n_ins = int(os.environ.get("N_INSURANCE", "12"))
    n_auth = int(os.environ.get("N_AUTH", "0"))
    man = generate_dataset(out_dir, n_med, n_ins, n_auth)
    print(json.dumps(man, indent=2))
