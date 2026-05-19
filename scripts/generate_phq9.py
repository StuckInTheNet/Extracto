"""Generate synthetic PHQ-9 depression screening forms.

PHQ-9 is a 9-item self-report questionnaire widely used to screen and measure
depression severity. It's commonly referenced in personal injury and disability
litigation as evidence of psychological damages.

Structure:
- Header + column labels
- 9 questions, each with a 4-option Likert row:
    0 = Not at all
    1 = Several days
    2 = More than half the days
    3 = Nearly every day
- Total score (sum of 9 items, 0-27)
- Functional difficulty question (PHQ-9 item #10)
- Optional clinician scoring / signature

Scoring interpretation:
    0-4   Minimal depression
    5-9   Mild
    10-14 Moderate
    15-19 Moderately severe
    20-27 Severe
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
MARGIN_L = 54
MARGIN_R = 54

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]

LIKERT_OPTIONS = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day",
]

DIFFICULTY_OPTIONS = [
    "Not difficult at all",
    "Somewhat difficult",
    "Very difficult",
    "Extremely difficult",
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Drew", "Morgan", "Riley", "Jamie", "Avery", "Cameron"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]

TITLES = [
    "Patient Health Questionnaire (PHQ-9)",
    "PHQ-9 Depression Screening",
    "Depression Screening Tool - PHQ-9",
]


def rand_date(start="2023-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    return d.strftime("%m/%d/%Y")


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def score_interpretation(total: int) -> str:
    if total <= 4:
        return "Minimal depression"
    if total <= 9:
        return "Mild depression"
    if total <= 14:
        return "Moderate depression"
    if total <= 19:
        return "Moderately severe depression"
    return "Severe depression"


# --- Drawing primitives ---

def draw_checkbox(c: canvas.Canvas, x, y, size=10, checked=False, style="check"):
    """Draw a checkbox at (x, y) with an optional check/X/filled mark."""
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
        elif style == "filled":
            c.setFillColor(colors.black)
            c.rect(x + 2, y + 2, size - 4, size - 4, stroke=0, fill=1)


def draw_header(c: canvas.Canvas, patient_name: str, date: str, title: str):
    """Draw the form header with patient info."""
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN_L, PAGE_H - 60, title)

    c.setFont("Helvetica", 10)
    c.drawString(MARGIN_L, PAGE_H - 85, f"Patient Name: {patient_name}")
    c.drawString(PAGE_W - MARGIN_R - 150, PAGE_H - 85, f"Date: {date}")

    c.setFont("Helvetica-Oblique", 9)
    instructions = "Over the last 2 weeks, how often have you been bothered by any of the following problems?"
    c.drawString(MARGIN_L, PAGE_H - 110, instructions)


def draw_column_headers(c: canvas.Canvas, y: float, col_xs: list[float], use_numeric: bool):
    """Draw the 4 Likert option column headers."""
    c.setFont("Helvetica-Bold", 7)
    for i, (label, cx) in enumerate(zip(LIKERT_OPTIONS, col_xs)):
        # Wrap long labels
        if use_numeric:
            c.drawCentredString(cx, y, str(i))
        else:
            # Two-line layout for long labels
            words = label.split()
            if len(words) <= 2:
                c.drawCentredString(cx, y, label)
            else:
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])
                c.drawCentredString(cx, y + 8, line1)
                c.drawCentredString(cx, y, line2)


def draw_question_row(
    c: canvas.Canvas,
    y: float,
    question_num: int,
    question_text: str,
    score: int,
    col_xs: list[float],
    checkbox_size: int = 10,
    mark_style: str = "check",
):
    """Draw a single numbered question with its Likert row of 4 checkboxes."""
    c.setFont("Helvetica", 9)
    # Question number
    c.drawString(MARGIN_L, y + 2, f"{question_num}.")
    # Question text with wrapping
    _draw_wrapped_text(c, MARGIN_L + 14, y + 2, question_text, col_xs[0] - MARGIN_L - 30, font_size=9)

    # 4 Likert checkboxes
    for i, cx in enumerate(col_xs):
        checked = (i == score)
        draw_checkbox(c, cx - checkbox_size / 2, y - 2, size=checkbox_size, checked=checked, style=mark_style)


def _draw_wrapped_text(c: canvas.Canvas, x: float, y: float, text: str, max_w: float, font_size=9):
    """Draw text with word wrapping, max 2 lines."""
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

    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]

    for i, line in enumerate(lines[:2]):
        c.drawString(x, y - i * (font_size + 1), line)


def draw_total_score(c: canvas.Canvas, y: float, total: int):
    """Draw the total score row."""
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, y + 12, PAGE_W - MARGIN_R, y + 12)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN_L, y, f"Total Score: {total}")
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L + 120, y, f"({score_interpretation(total)})")


def draw_difficulty_question(
    c: canvas.Canvas,
    y: float,
    difficulty: int,
    col_xs: list[float],
    use_numeric: bool,
    checkbox_size: int = 10,
    mark_style: str = "check",
):
    """Draw the functional difficulty question (item #10)."""
    c.setFont("Helvetica-Oblique", 9)
    prompt = (
        "If you checked off any problems, how difficult have these problems made "
        "it for you to do your work, take care of things at home, or get along with other people?"
    )
    _draw_wrapped_text(c, MARGIN_L, y + 18, prompt, PAGE_W - MARGIN_L - MARGIN_R, font_size=9)

    # Draw 4 column labels for difficulty
    c.setFont("Helvetica", 7)
    for label, cx in zip(DIFFICULTY_OPTIONS, col_xs):
        words = label.split()
        if len(words) <= 2:
            c.drawCentredString(cx, y - 12, label)
        else:
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            c.drawCentredString(cx, y - 8, line1)
            c.drawCentredString(cx, y - 16, line2)

    # Draw checkboxes
    for i, cx in enumerate(col_xs):
        checked = (i == difficulty)
        draw_checkbox(c, cx - checkbox_size / 2, y - 30, size=checkbox_size, checked=checked, style=mark_style)


def draw_signature(c: canvas.Canvas, y: float, clinician: str, date: str):
    """Draw a signature line at the bottom."""
    c.setFont("Helvetica", 9)
    c.setStrokeColor(colors.black)
    c.line(MARGIN_L, y, MARGIN_L + 200, y)
    c.line(PAGE_W - MARGIN_R - 150, y, PAGE_W - MARGIN_R, y)
    c.drawString(MARGIN_L, y - 12, f"Clinician: {clinician}")
    c.drawString(PAGE_W - MARGIN_R - 150, y - 12, f"Date: {date}")


# --- Scenario generation ---

def generate_scenario() -> dict[str, Any]:
    """Generate a random PHQ-9 scenario with 9 scores and metadata."""
    # Generate scores biased toward realistic depression distributions
    severity_profile = random.choices(
        ["minimal", "mild", "moderate", "severe"],
        weights=[0.25, 0.30, 0.25, 0.20],
    )[0]
    if severity_profile == "minimal":
        weights = [0.70, 0.20, 0.07, 0.03]
    elif severity_profile == "mild":
        weights = [0.40, 0.40, 0.15, 0.05]
    elif severity_profile == "moderate":
        weights = [0.20, 0.35, 0.30, 0.15]
    else:
        weights = [0.10, 0.20, 0.30, 0.40]

    scores = [random.choices([0, 1, 2, 3], weights=weights)[0] for _ in range(9)]
    total = sum(scores)

    # Difficulty roughly correlates with total
    if total < 5:
        difficulty = random.choices([0, 1, 2, 3], weights=[0.75, 0.20, 0.04, 0.01])[0]
    elif total < 10:
        difficulty = random.choices([0, 1, 2, 3], weights=[0.30, 0.50, 0.15, 0.05])[0]
    elif total < 15:
        difficulty = random.choices([0, 1, 2, 3], weights=[0.10, 0.35, 0.40, 0.15])[0]
    else:
        difficulty = random.choices([0, 1, 2, 3], weights=[0.05, 0.15, 0.35, 0.45])[0]

    return {
        "patient_name": rand_name(),
        "date": rand_date(),
        "clinician_name": f"Dr. {random.choice(LAST_NAMES)}",
        "title": random.choice(TITLES),
        "scores": scores,
        "total": total,
        "difficulty": difficulty,
        "use_numeric_headers": random.random() < 0.4,
        "mark_style": random.choice(["check", "x", "filled"]),
        "checkbox_size": random.choice([9, 10, 11, 12]),
    }


def generate_form(out_pdf: str, data: dict):
    """Render a complete PHQ-9 PDF."""
    c = canvas.Canvas(out_pdf, pagesize=letter)

    draw_header(c, data["patient_name"], data["date"], data["title"])

    # Column x positions for the 4 Likert options
    usable_w = PAGE_W - MARGIN_L - MARGIN_R
    question_col_w = usable_w * 0.55
    options_start = MARGIN_L + question_col_w
    options_w = usable_w - question_col_w
    col_xs = [options_start + (i + 0.5) * options_w / 4 for i in range(4)]

    # Column headers
    header_y = PAGE_H - 145
    draw_column_headers(c, header_y, col_xs, data["use_numeric_headers"])

    # 9 questions
    question_start_y = PAGE_H - 175
    row_h = 36
    for i, (question, score) in enumerate(zip(PHQ9_QUESTIONS, data["scores"])):
        row_y = question_start_y - i * row_h
        draw_question_row(
            c,
            row_y,
            i + 1,
            question,
            score,
            col_xs,
            checkbox_size=data["checkbox_size"],
            mark_style=data["mark_style"],
        )

    # Total score
    total_y = question_start_y - 9 * row_h - 10
    draw_total_score(c, total_y, data["total"])

    # Difficulty question
    diff_y = total_y - 45
    draw_difficulty_question(
        c,
        diff_y,
        data["difficulty"],
        col_xs,
        data["use_numeric_headers"],
        checkbox_size=data["checkbox_size"],
        mark_style=data["mark_style"],
    )

    # Signature
    sig_y = diff_y - 70
    if sig_y > 60:
        draw_signature(c, sig_y, data["clinician_name"], data["date"])

    c.showPage()
    c.save()


def ground_truth(data: dict) -> dict:
    return {
        "form_type": "phq9",
        "patient_name": data["patient_name"],
        "date": data["date"],
        "scores": data["scores"],
        "total": data["total"],
        "difficulty": data["difficulty"],
        "severity": score_interpretation(data["total"]),
    }


def generate_dataset(out_dir: str = "dataset/phq9", n: int = 25) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {"phq9": []}
    for i in range(1, n + 1):
        data = generate_scenario()
        pdf_path = out_path / f"phq9_{i:03d}.pdf"
        json_path = out_path / f"phq9_{i:03d}.json"
        generate_form(str(pdf_path), data)
        json_path.write_text(json.dumps(ground_truth(data), indent=2))
        manifest["phq9"].append({"pdf": str(pdf_path), "json": str(json_path)})

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    n = int(os.environ.get("N_FORMS", "25"))
    out_dir = os.environ.get("OUT_DIR", "dataset/phq9")
    manifest = generate_dataset(out_dir, n)
    print(f"Generated {len(manifest['phq9'])} PHQ-9 forms in {out_dir}")
