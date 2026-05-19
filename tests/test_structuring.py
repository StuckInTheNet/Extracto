"""Tests for form structuring modules — end-to-end extraction on generated forms."""

import json
from pathlib import Path

import pytest

from extracto.detection.controls import process_pdf
from extracto.structuring.forms import structure_page
from extracto.structuring.cms1500 import structure_cms1500
from extracto.structuring.phq9 import structure_phq9
from extracto.structuring.hipaa import structure_hipaa
from extracto.structuring.froi import structure_froi
from extracto.structuring.eob import structure_eob


def _load_and_extract(form_pdf, truth_json, structurer, **kwargs):
    """Helper: process a PDF + run structuring + load ground truth."""
    res = process_pdf(str(form_pdf), dpi=300)
    page = res["pages"][0]
    pred = structurer(page, **kwargs)
    truth = json.loads(Path(truth_json).read_text())
    return pred, truth


# --- Medical/Insurance ---


class TestMedicalStructuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/medical/medical_001.pdf")
        self.truth = Path("dataset/medical/medical_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_sex(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_page)
        assert pred.get("sex") is not None
        assert pred["sex"].lower() == truth["person"]["sex"].lower()

    def test_extracts_smoker(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_page)
        assert "Smoker" in pred
        assert pred["Smoker"] == truth["meta"]["smoker"]

    def test_extracts_allergies(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_page)
        assert "allergies" in pred
        assert isinstance(pred["allergies"], list)


# --- CMS-1500 ---


class TestCMS1500Structuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/cms1500/cms1500_001.pdf")
        self.truth = Path("dataset/cms1500/cms1500_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_insurance_type(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert pred.get("insurance_type") == truth["insurance_type"]

    def test_extracts_patient_sex(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert pred.get("patient_sex") == truth["patient_sex"]

    def test_extracts_conditions(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert pred.get("condition_employment") == truth["condition_employment"]
        assert pred.get("condition_auto") == truth["condition_auto"]
        assert pred.get("condition_other") == truth["condition_other"]

    def test_extracts_diagnoses(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert sorted(pred.get("diagnoses", [])) == sorted(truth["diagnoses"])

    def test_extracts_service_lines(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        pred_cpts = [s["cpt"] for s in pred.get("service_lines", [])]
        truth_cpts = [s["cpt"] for s in truth["service_lines"]]
        assert pred_cpts == truth_cpts

    def test_extracts_total_charge(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert abs(pred.get("total_charge", 0) - truth["total_charge"]) < 0.01

    def test_extracts_accept_assignment(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_cms1500)
        assert pred.get("accept_assignment") == truth["accept_assignment"]


# --- PHQ-9 ---


class TestPHQ9Structuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/phq9/phq9_001.pdf")
        self.truth = Path("dataset/phq9/phq9_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_all_9_scores(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_phq9, pdf_path=str(self.pdf)
        )
        assert pred.get("scores") == truth["scores"]

    def test_extracts_total(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_phq9, pdf_path=str(self.pdf)
        )
        assert pred.get("total") == truth["total"]

    def test_extracts_difficulty(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_phq9, pdf_path=str(self.pdf)
        )
        assert pred.get("difficulty") == truth["difficulty"]

    def test_checkbox_mode_on_synthetic(self):
        pred, _ = _load_and_extract(
            self.pdf, self.truth, structure_phq9, pdf_path=str(self.pdf)
        )
        assert pred.get("extraction_mode") == "checkboxes"


# --- HIPAA ---


class TestHIPAAStructuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/hipaa/hipaa_001.pdf")
        self.truth = Path("dataset/hipaa/hipaa_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_patient_name(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_hipaa)
        assert pred.get("patient_name") == truth["patient_name"]

    def test_extracts_date_range(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_hipaa)
        assert pred.get("date_range_from") == truth["date_range_from"]
        assert pred.get("date_range_to") == truth["date_range_to"]

    def test_extracts_excluded_categories(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_hipaa)
        assert sorted(pred.get("excluded_categories", [])) == sorted(truth["excluded_categories"])

    def test_extracts_purposes(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_hipaa)
        assert sorted(pred.get("purposes", [])) == sorted(truth["purposes"])

    def test_extracts_expiration_type(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_hipaa)
        assert pred.get("expiration_type") == truth["expiration_type"]


# --- FROI ---


class TestFROIStructuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/froi/froi_001.pdf")
        self.truth = Path("dataset/froi/froi_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_employee_name(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_froi, pdf_path=str(self.pdf)
        )
        assert pred.get("employee_name") == truth["employee_name"]

    def test_extracts_on_premises(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_froi, pdf_path=str(self.pdf)
        )
        assert pred.get("on_premises") == truth["on_premises"]

    def test_extracts_injured_body_parts(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_froi, pdf_path=str(self.pdf)
        )
        assert sorted(pred.get("injured_body_parts", [])) == sorted(truth["injured_body_parts"])

    def test_extracts_nature_of_injury(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_froi, pdf_path=str(self.pdf)
        )
        assert pred.get("nature_of_injury") == truth["nature_of_injury"]

    def test_extracts_causes(self):
        pred, truth = _load_and_extract(
            self.pdf, self.truth, structure_froi, pdf_path=str(self.pdf)
        )
        assert sorted(pred.get("causes", [])) == sorted(truth["causes"])


# --- EOB ---


class TestEOBStructuring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.pdf = Path("dataset/eob/eob_001.pdf")
        self.truth = Path("dataset/eob/eob_001.json")
        if not self.pdf.exists():
            pytest.skip("Dataset not generated")

    def test_extracts_payer_name(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert pred.get("payer_name") == truth["payer_name"]

    def test_extracts_claim_number(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert pred.get("claim_number") == truth["claim_number"]

    def test_extracts_service_line_count(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert pred.get("service_line_count") == truth["service_line_count"]

    def test_extracts_total_billed(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert abs(pred.get("total_billed", 0) - truth["total_billed"]) < 0.01

    def test_extracts_total_plan_paid(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert abs(pred.get("total_plan_paid", 0) - truth["total_plan_paid"]) < 0.01

    def test_extracts_reason_codes(self):
        pred, truth = _load_and_extract(self.pdf, self.truth, structure_eob)
        assert sorted(pred.get("reason_codes_used", [])) == sorted(truth["reason_codes_used"])
