"""Tests for PDF-native detection and mark detection."""

import json
from pathlib import Path

import fitz
import pytest

from extracto.detection.controls import process_pdf
from extracto.detection.pdf_native import (
    extract_controls_from_drawings,
    has_useful_drawings,
    _classify_drawing,
    _is_square,
)
from extracto.detection.marks import (
    Mark,
    find_marks,
    any_mark_overlapping,
    closest_mark,
    find_overlaid_text,
    nearest_overlaid_text,
)


# --- PDF-native detection ---


class TestIsSquare:
    def test_valid_square(self):
        assert _is_square(fitz.Rect(0, 0, 12, 12))

    def test_too_small(self):
        assert not _is_square(fitz.Rect(0, 0, 3, 3))

    def test_too_large(self):
        assert not _is_square(fitz.Rect(0, 0, 30, 30))

    def test_not_square(self):
        assert not _is_square(fitz.Rect(0, 0, 12, 50))

    def test_near_square(self):
        # Aspect diff of 2 is within the 3.0 tolerance
        assert _is_square(fitz.Rect(0, 0, 10, 12))


class TestClassifyDrawing:
    def test_rectangle_with_re_op(self):
        d = {"items": [("re", fitz.Rect(0, 0, 12, 12))]}
        assert _classify_drawing(d) == "rect"

    def test_circle_with_curves_only(self):
        d = {"items": [("c", 1, 2, 3, 4)] * 4}
        assert _classify_drawing(d) == "circle"

    def test_rounded_rect_with_mixed_ops(self):
        d = {"items": [("l", 1, 2), ("c", 1, 2, 3, 4)] * 4}
        assert _classify_drawing(d) == "rect"

    def test_empty_items(self):
        d = {"items": []}
        assert _classify_drawing(d) is None


class TestProcessPdf:
    """Integration tests using generated synthetic forms."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure dataset exists."""
        self.med_pdf = Path("dataset/medical/medical_001.pdf")
        self.cms_pdf = Path("dataset/cms1500/cms1500_001.pdf")
        if not self.med_pdf.exists() or not self.cms_pdf.exists():
            pytest.skip("Dataset not generated; run scripts/generate_benchmark_docs.py first")

    def test_medical_form_has_controls(self):
        res = process_pdf(str(self.med_pdf), dpi=300)
        assert len(res["pages"]) == 1
        assert len(res["pages"][0]["controls"]) > 10

    def test_medical_form_has_lines(self):
        res = process_pdf(str(self.med_pdf), dpi=300)
        assert len(res["pages"][0]["lines"]) > 20

    def test_cms1500_has_controls(self):
        res = process_pdf(str(self.cms_pdf), dpi=300)
        controls = res["pages"][0]["controls"]
        assert len(controls) > 15
        kinds = {c["kind"] for c in controls}
        assert "checkbox" in kinds

    def test_controls_have_required_fields(self):
        res = process_pdf(str(self.med_pdf), dpi=300)
        for c in res["pages"][0]["controls"]:
            assert "kind" in c
            assert "bbox" in c
            assert "selected" in c
            assert "conf" in c
            assert c["kind"] in ("checkbox", "radio")

    def test_native_detection_preferred_over_cv(self):
        """PDF-native detection should find controls without rasterizing."""
        doc = fitz.open(str(self.med_pdf))
        page = doc[0]
        assert has_useful_drawings(page)
        controls = extract_controls_from_drawings(page)
        assert len(controls) > 10
        doc.close()


# --- Mark detection ---


class TestMark:
    def test_overlaps_true(self):
        m = Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10)
        assert m.overlaps((12, 12, 18, 18))

    def test_overlaps_false(self):
        m = Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10)
        assert not m.overlaps((50, 50, 60, 60))

    def test_overlaps_with_margin(self):
        m = Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10)
        assert m.overlaps((21, 10, 30, 20), margin=2)

    def test_distance_to_point(self):
        m = Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10)
        assert m.distance_to_point(15, 15) == 0.0
        assert abs(m.distance_to_point(18, 19) - 5.0) < 0.1


class TestClosestMark:
    def test_finds_closest(self):
        marks = [
            Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10),
            Mark(bbox=(50, 50, 60, 60), center=(55, 55), color=(1, 0, 0), kind="stroke", size=10),
        ]
        result = closest_mark(marks, 52, 52)
        assert result is marks[1]

    def test_none_within_distance(self):
        marks = [
            Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10),
        ]
        assert closest_mark(marks, 100, 100, max_distance=5) is None


class TestAnyMarkOverlapping:
    def test_true(self):
        marks = [
            Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10),
        ]
        assert any_mark_overlapping(marks, (12, 12, 18, 18))

    def test_false(self):
        marks = [
            Mark(bbox=(10, 10, 20, 20), center=(15, 15), color=(1, 0, 0), kind="stroke", size=10),
        ]
        assert not any_mark_overlapping(marks, (50, 50, 60, 60))

    def test_empty_marks(self):
        assert not any_mark_overlapping([], (10, 10, 20, 20))
