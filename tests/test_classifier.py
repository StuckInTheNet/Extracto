"""Tests for form type classification."""

import pytest

from extracto.pipeline.classifier import classify_page, classify_page_from_lines


class TestClassifyPage:
    def test_cms1500(self):
        text = "HEALTH INSURANCE CLAIM FORM APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE"
        form_type, conf = classify_page(text)
        assert form_type == "cms1500"
        assert conf >= 0.5

    def test_phq9(self):
        text = "Patient Health Questionnaire PHQ-9 Little interest or pleasure"
        form_type, conf = classify_page(text)
        assert form_type == "phq9"
        assert conf >= 0.5

    def test_hipaa(self):
        text = "Authorization for Release of Protected Health Information Do NOT release"
        form_type, conf = classify_page(text)
        assert form_type == "hipaa"
        assert conf >= 0.5

    def test_froi(self):
        text = "FIRST REPORT OF INJURY Workers' Compensation Body Parts Injured"
        form_type, conf = classify_page(text)
        assert form_type == "froi"
        assert conf >= 0.3

    def test_eob(self):
        text = "EXPLANATION OF BENEFITS This is not a bill Plan Paid Claim Detail"
        form_type, conf = classify_page(text)
        assert form_type == "eob"
        assert conf >= 0.75

    def test_medical_intake(self):
        text = "Patient Intake Form Allergies (check all Current Symptoms"
        form_type, conf = classify_page(text)
        assert form_type == "medical"
        assert conf >= 0.4

    def test_insurance_claim(self):
        text = "Insurance Claim Form Claim Type Is this work-related"
        form_type, conf = classify_page(text)
        assert form_type == "insurance"
        assert conf >= 0.4

    def test_unknown_text(self):
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        form_type, conf = classify_page(text)
        assert form_type == "unknown"
        assert conf == 0.0

    def test_empty_text(self):
        form_type, conf = classify_page("")
        assert form_type == "unknown"

    def test_cms1500_beats_generic_insurance(self):
        """CMS-1500 is also a 'health insurance claim' but should win on specificity."""
        text = "HEALTH INSURANCE CLAIM FORM NUCC 1a. INSURED Claim Type"
        form_type, _ = classify_page(text)
        assert form_type == "cms1500"


class TestClassifyFromLines:
    def test_from_line_dicts(self):
        lines = [
            {"text": "EXPLANATION OF BENEFITS"},
            {"text": "This is not a bill"},
            {"text": "Claim Detail"},
        ]
        form_type, conf = classify_page_from_lines(lines)
        assert form_type == "eob"
