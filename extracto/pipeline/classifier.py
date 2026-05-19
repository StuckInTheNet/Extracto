"""Form type classifier — determine what kind of form a PDF page is.

Uses text anchor patterns to classify pages. Each form type has distinctive
header or section text that uniquely identifies it. The classifier extracts
the first ~500 characters of text from a page and pattern-matches against
known form signatures.

Supported form types:
- cms1500: "HEALTH INSURANCE CLAIM FORM", "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE"
- phq9: "PHQ-9", "Patient Health Questionnaire"
- hipaa: "Authorization for Release", "Protected Health Information"
- froi: "FIRST REPORT OF INJURY", "WORKERS' COMPENSATION"
- eob: "EXPLANATION OF BENEFITS"
- medical: "Patient Intake", "Medical Intake", "New Patient Information"
- insurance: "Insurance Claim", "Claim Submission"
"""

from __future__ import annotations

import re
from typing import Any


# Each entry: (form_type, list_of_patterns, priority)
# Priority breaks ties if multiple patterns match — higher wins.
# Patterns are checked against the full page text (case-insensitive).
FORM_SIGNATURES: list[tuple[str, list[str], int]] = [
    (
        "cms1500",
        [
            "health insurance claim form",
            "approved by national uniform claim committee",
            "nucc",
            "1a. insured",
        ],
        100,  # High priority — very distinctive
    ),
    (
        "phq9",
        [
            "phq-9",
            "phq 9",
            "patient health questionnaire",
            "little interest or pleasure",
        ],
        90,
    ),
    (
        "hipaa",
        [
            "authorization for release of protected health",
            "hipaa authorization",
            "specially protected information",
            "do not release",
        ],
        85,
    ),
    (
        "froi",
        [
            "first report of injury",
            "workers' compensation",
            "workers compensation",
            "employer's first report",
            "body parts injured",
            "workers' comp claim",
            "workers compensation claim",
            "work-related injury",
        ],
        80,
    ),
    (
        "eob",
        [
            "explanation of benefits",
            "this is not a bill",
            "plan paid",
            "claim detail",
        ],
        75,
    ),
    (
        "medical",
        [
            "patient intake form",
            "medical intake",
            "new patient information",
            "current symptoms",
            "allergies (check all",
            "allergies",
            "symptoms",
            "smoker",
            "diabetic",
        ],
        60,
    ),
    (
        "insurance",
        [
            "insurance claim form",
            "health insurance claim",
            "claim submission",
            "claim type",
            "is this work-related",
            "auto accident",
            "policy holder",
            "services rendered",
        ],
        50,  # Lower than CMS-1500 which also says "health insurance claim"
    ),
]


def classify_page(page_text: str) -> tuple[str, float]:
    """Classify a page's form type from its extracted text.

    Args:
        page_text: All text from the page (concatenated lines, case doesn't matter).

    Returns:
        (form_type, confidence) where confidence is 0.0-1.0 based on how many
        signature patterns matched.
    """
    # Normalize Unicode: curly/smart quotes → ASCII so pattern matching works
    # on government PDFs that use typographic apostrophes
    text_lower = page_text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')

    scores: list[tuple[str, float, int]] = []
    for form_type, patterns, priority in FORM_SIGNATURES:
        matched = sum(1 for p in patterns if p in text_lower)
        if matched > 0:
            # Confidence scales with matches but floors at 0.7 for any match,
            # since even a single strong anchor (e.g. "Patient Intake Form")
            # is a reliable signal. Additional matches push toward 1.0.
            raw = matched / len(patterns)
            confidence = 0.7 + 0.3 * raw
            scores.append((form_type, confidence, priority))

    if not scores:
        return ("unknown", 0.0)

    # Sort by: confidence desc, then priority desc
    scores.sort(key=lambda s: (s[1], s[2]), reverse=True)
    best_type, best_conf, _ = scores[0]
    return (best_type, round(best_conf, 3))


def classify_page_from_lines(lines: list[dict[str, Any]]) -> tuple[str, float]:
    """Classify from structured lines (as returned by process_pdf)."""
    text = " ".join(ln.get("text", "") for ln in lines)
    return classify_page(text)


def classify_pdf(pdf_path: str) -> list[tuple[int, str, float]]:
    """Classify all pages in a PDF.

    Returns list of (page_index, form_type, confidence) for each page.
    """
    import fitz

    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        text = page.get_text()
        form_type, conf = classify_page(text)
        results.append((i, form_type, conf))
    doc.close()
    return results
