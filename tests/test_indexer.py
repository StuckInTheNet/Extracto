"""Tests for the medical records indexer."""

import json
from pathlib import Path

import fitz
import pytest

from extracto.pipeline.indexer import (
    analyze_page,
    detect_boundaries,
    _extract_dos,
    _classify_doc_type,
    PageInfo,
)


class TestExtractDOS:
    def test_finds_date_near_dos_keyword(self):
        # Realistic spacing: DOS and DOB on separate lines with other text between
        text = "Mitchell Family Medicine\nDate of Service: 03/14/2024\nPatient: John Doe\nDate of Birth: 07/22/1985"
        dos = _extract_dos(text, None)
        assert dos == "03/14/2024"

    def test_prefers_dos_over_dob(self):
        text = "Central Neurology\nDate of Birth: 07/22/1985\nMRN: 12345\nVisit Date: 11/15/2024\nReason for visit: headache"
        dos = _extract_dos(text, None)
        assert dos == "11/15/2024"

    def test_rejects_only_dob(self):
        text = "Date of Birth: 07/22/1985\nNo visit date here."
        dos = _extract_dos(text, None)
        assert dos is None

    def test_no_dates(self):
        dos = _extract_dos("No dates in this text at all.", None)
        assert dos is None

    def test_date_near_top_preferred(self):
        text = "Date: 01/15/2025\n" + "x " * 300 + "Date: 06/30/2025"
        dos = _extract_dos(text, None)
        assert dos == "01/15/2025"


class TestClassifyDocType:
    def test_office_visit(self):
        assert _classify_doc_type("Progress Note for follow-up visit") == "Office Visit"

    def test_er_report(self):
        assert _classify_doc_type("Emergency Department ER Report") == "ER Report"

    def test_operative(self):
        assert _classify_doc_type("Operative Report for lumbar procedure") == "Operative Report"

    def test_radiology(self):
        assert _classify_doc_type("MRI Lumbar Spine Imaging Report") == "Imaging Report"

    def test_pt_note(self):
        assert _classify_doc_type("Physical Therapy PT Session Note") == "PT Note"

    def test_lab(self):
        assert _classify_doc_type("Quest Diagnostics Lab Result CBC") == "Lab Result"

    def test_default(self):
        assert _classify_doc_type("Some generic text") == "Clinical Note"


class TestDetectBoundaries:
    def _make_page(self, num, provider=None, dos=None, doc_type=None,
                   is_sep=False, is_cover=False):
        return PageInfo(
            page_num=num,
            header_text="",
            full_text="",
            provider_name=provider,
            dos=dos,
            doc_type=doc_type,
            is_separator=is_sep,
            is_cover=is_cover,
        )

    def test_single_page(self):
        pages = [self._make_page(1, "DrA", "01/01/2025", "Office Visit")]
        segs = detect_boundaries(pages)
        assert len(segs) == 1
        assert segs[0].provider == "DrA"

    def test_splits_on_provider_change(self):
        pages = [
            self._make_page(1, "DrA", "01/01/2025", "Office Visit"),
            self._make_page(2, "DrA", "01/01/2025", "Office Visit"),
            self._make_page(3, "DrB", "02/01/2025", "Lab Result"),
        ]
        segs = detect_boundaries(pages)
        clinical = [s for s in segs if s.doc_type not in ("Cover Sheet", "Separator")]
        assert len(clinical) == 2
        assert clinical[0].provider == "DrA"
        assert clinical[1].provider == "DrB"

    def test_splits_on_dos_change(self):
        pages = [
            self._make_page(1, "DrA", "01/01/2025", "Office Visit"),
            self._make_page(2, "DrA", "02/15/2025", "Office Visit"),
        ]
        segs = detect_boundaries(pages)
        assert len(segs) == 2
        assert segs[0].dos == "01/01/2025"
        assert segs[1].dos == "02/15/2025"

    def test_separator_creates_boundary(self):
        pages = [
            self._make_page(1, "DrA", "01/01/2025", "Office Visit"),
            self._make_page(2, is_sep=True),
            self._make_page(3, "DrB", "02/01/2025", "Office Visit"),
        ]
        segs = detect_boundaries(pages)
        clinical = [s for s in segs if s.doc_type not in ("Separator", "Unknown")]
        assert len(clinical) == 2

    def test_continuation_pages_grouped(self):
        pages = [
            self._make_page(1, "DrA", "01/01/2025", "Office Visit"),
            self._make_page(2, None, None, None),  # continuation - no header
            self._make_page(3, None, None, None),
        ]
        segs = detect_boundaries(pages)
        assert len(segs) == 1
        assert segs[0].start_page == 1
        assert segs[0].end_page == 3
        assert segs[0].page_count == 3


class TestIndexerIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bundle = Path("dataset/medical_records/records_bundle.pdf")
        self.gt = Path("dataset/medical_records/ground_truth.json")
        if not self.bundle.exists():
            pytest.skip("Records bundle not generated")

    def test_all_providers_found(self):
        """All ground truth practice names must appear in the index."""
        from extracto.pipeline.indexer import build_index
        idx = build_index(str(self.bundle))
        gt = json.loads(self.gt.read_text())
        gt_providers = set(
            d["provider"] for d in gt["documents"]
            if d["doc_type"] not in ("cover_sheet", "separator")
        )
        idx_providers = set(idx["providers"].keys())
        # Every GT provider must be in the index (index may have extras from cover sheets)
        assert gt_providers.issubset(idx_providers), f"Missing: {gt_providers - idx_providers}"

    def test_encounter_count_reasonable(self):
        """Index should find a similar number of encounters as ground truth."""
        from extracto.pipeline.indexer import build_index
        idx = build_index(str(self.bundle))
        gt = json.loads(self.gt.read_text())
        gt_encounters = len([d for d in gt["documents"] if d["doc_type"] not in ("cover_sheet", "separator")])
        # The indexer may over-split or under-split slightly; allow 40% tolerance
        # since page counting in the generator vs actual pages can differ
        assert idx["encounter_count"] >= gt_encounters * 0.6
        assert idx["encounter_count"] <= gt_encounters * 1.5
