"""Tests for the generic document extractor."""

import json
from pathlib import Path

import pytest

from extracto.structuring.generic import (
    extract_key_value_pairs,
    extract_selected_controls,
    extract_dates_with_context,
    extract_entities,
    extract_tables,
    structure_generic,
)
from extracto.detection.controls import process_pdf


# --- Key-value pair extraction ---


class TestKeyValuePairs:
    def test_simple_kv(self):
        lines = [{"text": "Patient Name: Jordan Smith"}]
        pairs = extract_key_value_pairs(lines)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "Patient Name"
        assert pairs[0]["value"] == "Jordan Smith"

    def test_multiple_kv(self):
        lines = [
            {"text": "Date of Birth: 07/22/1985"},
            {"text": "Phone: (555) 123-4567"},
            {"text": "Claim Number: CLM-2024-89012"},
        ]
        pairs = extract_key_value_pairs(lines)
        assert len(pairs) == 3
        labels = [p["label"] for p in pairs]
        assert "Date of Birth" in labels
        assert "Phone" in labels
        assert "Claim Number" in labels

    def test_skips_blank_underline_values(self):
        lines = [{"text": "Name: ______________________"}]
        pairs = extract_key_value_pairs(lines)
        assert len(pairs) == 0

    def test_skips_lowercase_labels(self):
        lines = [{"text": "this is just regular text that has a colon: here"}]
        pairs = extract_key_value_pairs(lines)
        assert len(pairs) == 0

    def test_handles_labels_with_special_chars(self):
        lines = [{"text": "Patient's Name: Jordan Smith"}]
        pairs = extract_key_value_pairs(lines)
        assert len(pairs) == 1
        assert pairs[0]["label"] == "Patient's Name"


# --- Selected controls ---


class TestSelectedControls:
    def test_returns_selected_only(self):
        controls = [
            {"kind": "checkbox", "selected": True, "label": "Male", "conf": 0.95},
            {"kind": "checkbox", "selected": False, "label": "Female", "conf": 0.95},
            {"kind": "radio", "selected": True, "label": "Yes", "conf": 0.98},
        ]
        selected = extract_selected_controls(controls)
        assert len(selected) == 2
        labels = [s["label"] for s in selected]
        assert "Male" in labels
        assert "Yes" in labels
        assert "Female" not in labels

    def test_empty_controls(self):
        assert extract_selected_controls([]) == []


# --- Dates with context ---


class TestDatesWithContext:
    def test_finds_date_with_label(self):
        lines = [{"text": "Date of Injury: 09/14/2024"}]
        dates = extract_dates_with_context(lines)
        assert len(dates) == 1
        assert dates[0]["date"] == "09/14/2024"
        assert dates[0]["context"] == "Date of Injury"

    def test_finds_multiple_dates(self):
        lines = [
            {"text": "Date of Birth: 07/22/1985"},
            {"text": "Visit Date: 03/14/2025"},
        ]
        dates = extract_dates_with_context(lines)
        assert len(dates) == 2

    def test_deduplicates_dates(self):
        lines = [
            {"text": "Date: 09/14/2024"},
            {"text": "Injury on 09/14/2024"},
        ]
        dates = extract_dates_with_context(lines)
        assert len(dates) == 1

    def test_no_dates(self):
        lines = [{"text": "No dates in this text"}]
        assert extract_dates_with_context(lines) == []


# --- Entity extraction ---


class TestEntities:
    def test_phone_numbers(self):
        entities = extract_entities("Call us at (312) 555-8901 or 800-555-1234")
        assert "phone_numbers" in entities
        assert len(entities["phone_numbers"]) == 2

    def test_icd10_codes(self):
        entities = extract_entities("Diagnosis: M54.2, S13.4XXA, cervicalgia")
        assert "icd10_codes" in entities
        assert "M54.2" in entities["icd10_codes"]
        assert "S13.4XXA" in entities["icd10_codes"]

    def test_npi(self):
        entities = extract_entities("NPI: 1234567890")
        assert "npi" in entities
        assert "1234567890" in entities["npi"]

    def test_currency(self):
        entities = extract_entities("Total: $4,095.00, Subtotal: $3,200.00")
        assert "currency_values" in entities
        assert "$4,095.00" in entities["currency_values"]
        # Should be sorted by value descending
        assert entities["currency_values"][0] == "$4,095.00"

    def test_ssn_masking(self):
        entities = extract_entities("SSN: 123-45-6789")
        assert "ssn_detected" in entities
        assert entities["ssn_detected"][0] == "***-**-6789"

    def test_empty_text(self):
        entities = extract_entities("Nothing extractable here")
        assert entities == {}


# --- Table detection ---


class TestTableDetection:
    def test_detects_simple_table(self):
        lines = [
            {"text": "Name", "bbox": [10, 100, 60, 110]},
            {"text": "Date", "bbox": [100, 100, 140, 110]},
            {"text": "Amount", "bbox": [200, 100, 260, 110]},
            {"text": "Smith", "bbox": [10, 114, 60, 124]},
            {"text": "01/01", "bbox": [100, 114, 140, 124]},
            {"text": "$100", "bbox": [200, 114, 260, 124]},
            {"text": "Jones", "bbox": [10, 128, 60, 138]},
            {"text": "02/01", "bbox": [100, 128, 140, 138]},
            {"text": "$200", "bbox": [200, 128, 260, 138]},
        ]
        tables = extract_tables(lines)
        assert len(tables) == 1
        assert tables[0]["row_count"] == 3
        assert tables[0]["col_count"] == 3
        assert tables[0]["header"] == ["Name", "Date", "Amount"]
        assert len(tables[0]["rows"]) == 2

    def test_no_table_in_single_column(self):
        lines = [
            {"text": "Line 1", "bbox": [10, 100, 60, 110]},
            {"text": "Line 2", "bbox": [10, 114, 60, 124]},
            {"text": "Line 3", "bbox": [10, 128, 60, 138]},
        ]
        tables = extract_tables(lines)
        assert len(tables) == 0  # Single column = not a table

    def test_empty_lines(self):
        assert extract_tables([]) == []


# --- End-to-end generic extraction ---


class TestStructureGeneric:
    def test_returns_form_type_generic(self):
        page = {"controls": [], "lines": []}
        result = structure_generic(page)
        assert result["form_type"] == "generic"
        assert result["extraction_mode"] == "generic"

    def test_extracts_stats(self):
        page = {
            "controls": [
                {"kind": "checkbox", "selected": True, "label": "Yes", "conf": 0.9},
            ],
            "lines": [
                {"text": "Patient Name: Test User"},
                {"text": "Date: 01/15/2025"},
            ],
        }
        result = structure_generic(page)
        assert result["stats"]["text_lines"] == 2
        assert result["stats"]["controls_selected"] == 1

    def test_promotes_patient_name(self):
        page = {
            "controls": [],
            "lines": [{"text": "Patient Name: Jordan Smith"}],
        }
        result = structure_generic(page)
        assert result.get("patient_name") == "Jordan Smith"

    def test_promotes_patient_dob(self):
        page = {
            "controls": [],
            "lines": [{"text": "Date of Birth: 07/22/1985"}],
        }
        result = structure_generic(page)
        assert result.get("patient_dob") == "07/22/1985"


class TestGenericOnUnknownForms:
    """Integration tests using generated unknown forms."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.lor = Path("dataset/unknown_forms/lor_001.pdf")
        self.prop = Path("dataset/unknown_forms/property_damage_001.pdf")
        self.disability = Path("dataset/unknown_forms/disability_cert_001.pdf")
        if not self.lor.exists():
            pytest.skip("Unknown forms not generated; run scripts/generate_unknown_forms.py")

    def test_lor_extracts_client_name(self):
        res = process_pdf(str(self.lor), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.lor))
        # The LOR should have key-value pairs with "Our Client"
        kv_labels = [p["label"] for p in result.get("key_value_pairs", [])]
        assert "Our Client" in kv_labels

    def test_lor_extracts_dates(self):
        res = process_pdf(str(self.lor), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.lor))
        date_strs = [d["date"] for d in result.get("dates", [])]
        assert "09/14/2024" in date_strs  # Date of Injury

    def test_property_damage_finds_currency(self):
        res = process_pdf(str(self.prop), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.prop))
        currencies = result.get("entities", {}).get("currency_values", [])
        assert "$4,095.00" in currencies

    def test_disability_finds_icd10(self):
        res = process_pdf(str(self.disability), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.disability))
        codes = result.get("entities", {}).get("icd10_codes", [])
        assert "M51.26" in codes
        assert "S13.4XXA" in codes

    def test_disability_finds_npi(self):
        res = process_pdf(str(self.disability), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.disability))
        npis = result.get("entities", {}).get("npi", [])
        assert "1234567890" in npis

    def test_disability_finds_section_headers(self):
        res = process_pdf(str(self.disability), dpi=300)
        result = structure_generic(res["pages"][0], pdf_path=str(self.disability))
        headers = result.get("section_headers", [])
        assert "ATTENDING PHYSICIAN STATEMENT" in headers
