"""Auto-extraction pipeline: classify form type and route to the right extractor.

This is the "throw everything at it" entrypoint. Given a directory of mixed
PDFs (or a single multi-page PDF), it:
1. Classifies each page by form type
2. Routes to the appropriate structuring module
3. Returns structured data for each form with its detected type
"""

from __future__ import annotations

import json
import csv
import time
from pathlib import Path
from typing import Any

from extracto.detection.controls import process_pdf
from extracto.pipeline.classifier import classify_page_from_lines
from extracto.structuring.cms1500 import structure_cms1500
from extracto.structuring.eob import structure_eob
from extracto.structuring.forms import structure_page
from extracto.structuring.froi import structure_froi
from extracto.structuring.generic import structure_generic
from extracto.structuring.hipaa import structure_hipaa
from extracto.structuring.phq9 import structure_phq9


STRUCTURERS = {
    "cms1500": lambda page, path: structure_cms1500(page, pdf_path=path),
    "phq9": lambda page, path: structure_phq9(page, pdf_path=path),
    "hipaa": lambda page, path: structure_hipaa(page, pdf_path=path),
    "froi": lambda page, path: structure_froi(page, pdf_path=path),
    "eob": lambda page, path: structure_eob(page, pdf_path=path),
    "medical": lambda page, path: structure_page(page, pdf_path=path),
    "insurance": lambda page, path: structure_page(page, pdf_path=path),
    "unknown": lambda page, path: structure_generic(page, pdf_path=path),
}


def auto_extract_single(pdf_path: str) -> dict[str, Any]:
    """Classify and extract a single PDF.

    Returns:
        {
            "file": str,
            "classified_type": str,
            "classification_confidence": float,
            "extraction": dict (the structured output),
            "processing_time_ms": float,
        }
    """
    start = time.monotonic()

    res = process_pdf(pdf_path, dpi=300, use_yolo=True)
    if not res["pages"]:
        return {
            "file": pdf_path,
            "classified_type": "unknown",
            "classification_confidence": 0.0,
            "extraction": {},
            "error": "No pages",
        }

    page = res["pages"][0]
    form_type, confidence = classify_page_from_lines(page.get("lines", []))

    # Minimum confidence: below this threshold, treat as unknown even if
    # a pattern partially matched. Prevents false positives like an ACORD
    # certificate matching "froi" because it mentions workers' comp coverage.
    MIN_CONFIDENCE = 0.15
    if confidence < MIN_CONFIDENCE:
        form_type = "unknown"

    structurer = STRUCTURERS.get(form_type, STRUCTURERS["unknown"])
    extraction = structurer(page, pdf_path)

    # Check if the specific extractor returned useful data
    useful_keys = {k for k in extraction if k not in (
        "form_type", "extraction_mode", "stats", "enriched_via_generic",
        "marks_detected", "overlays_detected"
    ) and extraction[k] is not None}
    extraction_empty = len(useful_keys) == 0

    # Always enrich with generic extraction to maximize field coverage
    is_ocr = page.get("text_source") == "ocr"
    needs_enrichment = True

    if needs_enrichment:
        generic = structure_generic(page, pdf_path=pdf_path)
        # Merge all generic fields that the specific extractor didn't produce
        for key in ("key_value_pairs", "dates", "entities", "tables",
                     "section_headers", "stats", "acroform_fields",
                     "patient_name", "patient_dob", "provider_name",
                     "mrn", "diagnoses", "selected_controls"):
            if key in generic and key not in extraction:
                extraction[key] = generic[key]
        if form_type != "unknown":
            extraction["enriched_via_generic"] = True

    elapsed = (time.monotonic() - start) * 1000

    return {
        "file": pdf_path,
        "classified_type": form_type,
        "classification_confidence": confidence,
        "extraction": extraction,
        "processing_time_ms": round(elapsed, 1),
    }


def auto_extract_batch(
    input_path: str,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Classify and extract all PDFs in a directory.

    Args:
        input_path: Directory containing mixed PDFs
        out_dir: Optional output directory for per-file JSON results

    Returns:
        Summary with classification breakdown, timing stats, and per-file results.
    """
    p = Path(input_path)
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.rglob("*.pdf"))

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    total_ms = 0.0
    errors = 0

    for f in files:
        result = auto_extract_single(str(f))
        results.append(result)

        ft = result["classified_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1
        total_ms += result.get("processing_time_ms", 0)
        if result.get("error"):
            errors += 1

        if out_dir:
            out_path = Path(out_dir) / f"{f.stem}_auto.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))

    summary = {
        "total_files": len(files),
        "classification_breakdown": dict(sorted(type_counts.items())),
        "errors": errors,
        "total_time_ms": round(total_ms, 1),
        "avg_time_ms": round(total_ms / len(files), 1) if files else 0,
    }

    if out_dir:
        # Write summary
        Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=2))

        # Write CSV for quick review
        csv_path = Path(out_dir, "results.csv")
        with open(csv_path, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=[
                "file", "classified_type", "confidence", "time_ms", "error"
            ])
            w.writeheader()
            for r in results:
                w.writerow({
                    "file": Path(r["file"]).name,
                    "classified_type": r["classified_type"],
                    "confidence": r["classification_confidence"],
                    "time_ms": r["processing_time_ms"],
                    "error": r.get("error", ""),
                })

    return {"summary": summary, "results": results}
