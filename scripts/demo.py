#!/usr/bin/env python3
"""Extracto Demo — see it work in 30 seconds.

Generates sample forms, runs auto-extraction, and prints results.

Usage:
    python scripts/demo.py
"""

import json
import sys
import time
from pathlib import Path


def main():
    print("=" * 60)
    print("EXTRACTO DEMO")
    print("=" * 60)
    print()

    # Step 1: Generate sample forms
    print("Step 1: Generating sample forms...")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from scripts.generate_benchmark_docs import generate_dataset

    demo_dir = Path("demo_output")
    demo_dir.mkdir(exist_ok=True)
    dataset_dir = demo_dir / "forms"

    generate_dataset(str(dataset_dir), n_medical=3, n_insurance=3)

    from scripts.generate_cms1500 import generate_dataset as gen_cms
    gen_cms(str(dataset_dir / "cms1500"), n=2)

    from scripts.generate_phq9 import generate_dataset as gen_phq9
    gen_phq9(str(dataset_dir / "phq9"), n=2)

    from scripts.generate_hipaa import generate_dataset as gen_hipaa
    gen_hipaa(str(dataset_dir / "hipaa"), n=2)

    from scripts.generate_froi import generate_dataset as gen_froi
    gen_froi(str(dataset_dir / "froi"), n=2)

    from scripts.generate_eob import generate_dataset as gen_eob
    gen_eob(str(dataset_dir / "eob"), n=2)

    # Collect all PDFs
    all_pdfs = sorted(dataset_dir.rglob("*.pdf"))
    print(f"  Generated {len(all_pdfs)} sample forms across 7 types")
    print()

    # Step 2: Auto-classify and extract
    print("Step 2: Auto-classifying and extracting...")
    start = time.monotonic()

    from extracto.pipeline.auto import auto_extract_single

    results = []
    for pdf in all_pdfs:
        result = auto_extract_single(str(pdf))
        results.append(result)

    elapsed = time.monotonic() - start
    print(f"  Processed {len(results)} forms in {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/form)")
    print()

    # Step 3: Show results
    print("Step 3: Results")
    print("-" * 60)

    # Group by type
    by_type = {}
    for r in results:
        t = r["classified_type"]
        by_type.setdefault(t, []).append(r)

    for form_type in sorted(by_type):
        items = by_type[form_type]
        print(f"\n  {form_type.upper()} ({len(items)} forms)")
        for r in items[:2]:  # Show first 2
            ext = r["extraction"]
            name = Path(r["file"]).name
            conf = r["classification_confidence"]
            ms = r["processing_time_ms"]

            # Pick the most interesting fields to show
            highlights = []
            for key in ["patient_name", "member_name", "employee_name",
                        "insurance_type", "payer_name",
                        "sex", "patient_sex", "employee_sex",
                        "Smoker", "Diabetic", "on_premises",
                        "diagnoses", "scores", "total", "total_charge",
                        "allergies", "symptoms",
                        "excluded_categories", "purposes",
                        "injured_body_parts", "nature_of_injury",
                        "claim_number", "reason_codes_used",
                        "condition_auto", "accept_assignment"]:
                val = ext.get(key)
                if val is not None:
                    vs = str(val)
                    if len(vs) > 40:
                        vs = vs[:37] + "..."
                    highlights.append(f"{key}={vs}")
                    if len(highlights) >= 5:
                        break

            print(f"    {name} (conf={conf}, {ms:.0f}ms)")
            for h in highlights:
                print(f"      {h}")

    # Step 4: Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    types = sorted(by_type.keys())
    print(f"  Form types detected: {', '.join(types)}")
    print(f"  Total forms:         {len(results)}")
    print(f"  Total time:          {elapsed:.1f}s")
    print(f"  Avg per form:        {elapsed/len(results)*1000:.0f}ms")
    print(f"  Errors:              {sum(1 for r in results if r.get('error'))}")
    print()
    print("  Output written to: demo_output/")
    print()
    print("  Try with your own PDFs:")
    print("    extracto auto your_forms/ --out results/")
    print("    extracto index medical_records.pdf --out index/")
    print()

    # Write results JSON
    out_file = demo_dir / "demo_results.json"
    with open(out_file, "w") as f:
        json.dump([{
            "file": Path(r["file"]).name,
            "type": r["classified_type"],
            "confidence": r["classification_confidence"],
            "time_ms": r["processing_time_ms"],
            "fields": {k: v for k, v in r["extraction"].items()
                       if k != "form_type" and v is not None},
        } for r in results], f, indent=2, default=str)
    print(f"  Full results: {out_file}")


if __name__ == "__main__":
    main()
