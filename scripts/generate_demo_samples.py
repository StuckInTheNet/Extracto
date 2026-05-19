"""Generate 800 sample documents for the Extracto demo site.

Spread across all form types:
  - CMS-1500:         120
  - EOB:              120
  - PHQ-9:            110
  - HIPAA:            110
  - FROI:             110
  - Medical Intake:   115
  - Insurance Claim:  115
  Total:              800
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_cms1500 import generate_dataset as gen_cms
from scripts.generate_eob import generate_dataset as gen_eob
from scripts.generate_phq9 import generate_dataset as gen_phq9
from scripts.generate_hipaa import generate_dataset as gen_hipaa
from scripts.generate_froi import generate_dataset as gen_froi
from scripts.generate_benchmark_docs import generate_dataset as gen_bench

OUT = "demo_samples"

def main():
    os.makedirs(OUT, exist_ok=True)

    tasks = [
        ("CMS-1500", lambda: gen_cms(f"{OUT}/cms1500", n=120)),
        ("EOB",      lambda: gen_eob(f"{OUT}/eob", n=120)),
        ("PHQ-9",    lambda: gen_phq9(f"{OUT}/phq9", n=110)),
        ("HIPAA",    lambda: gen_hipaa(f"{OUT}/hipaa", n=110)),
        ("FROI",     lambda: gen_froi(f"{OUT}/froi", n=110)),
        ("Medical/Insurance", lambda: gen_bench(f"{OUT}", n_medical=115, n_insurance=115, n_auth=0)),
    ]

    total = 0
    for name, fn in tasks:
        print(f"Generating {name}...", end=" ", flush=True)
        result = fn()
        count = result.get("count", 0) if isinstance(result, dict) else 0
        # Count files if result doesn't report
        if count == 0:
            for root, dirs, files in os.walk(OUT):
                count = sum(1 for f in files if f.endswith(".pdf"))
            count = count - total
        total += count
        print(f"{count} files")

    # Final count
    pdf_count = 0
    for root, dirs, files in os.walk(OUT):
        pdf_count += sum(1 for f in files if f.endswith(".pdf"))
    print(f"\nTotal: {pdf_count} PDFs in {OUT}/")


if __name__ == "__main__":
    main()
