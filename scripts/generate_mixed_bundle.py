"""Generate a single large PDF containing many forms jumbled together.

Usage:
    python scripts/generate_mixed_bundle.py [--count 500] [--output demo_samples/mixed_bundle_500.pdf]

Randomly selects forms from demo_samples/ and concatenates them in random order
into one monolithic PDF — simulating the kind of document dump a law firm receives.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import fitz


def build_mixed_bundle(
    source_dir: str = "demo_samples",
    count: int = 500,
    output: str | None = None,
    seed: int = 42,
) -> str:
    random.seed(seed)

    # Collect all source PDFs
    source_pdfs = sorted(Path(source_dir).rglob("*.pdf"))
    if not source_pdfs:
        raise FileNotFoundError(f"No PDFs found in {source_dir}")

    # Pick `count` forms at random (with replacement if needed)
    picks = [random.choice(source_pdfs) for _ in range(count)]
    random.shuffle(picks)

    if output is None:
        output = str(Path(source_dir) / f"mixed_bundle_{count}.pdf")

    print(f"Building mixed bundle: {count} forms from {len(source_pdfs)} source files")

    bundle = fitz.open()
    form_types = {}

    for i, pdf_path in enumerate(picks):
        try:
            src = fitz.open(str(pdf_path))
            bundle.insert_pdf(src)
            src.close()

            # Track types for summary
            ftype = pdf_path.parent.name
            form_types[ftype] = form_types.get(ftype, 0) + 1
        except Exception as e:
            print(f"  Warning: skipped {pdf_path.name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{count} forms added...")

    bundle.save(output)
    bundle.close()

    print(f"\nSaved: {output}")
    print(f"Total pages: {count}")
    print(f"Breakdown:")
    for ftype, n in sorted(form_types.items(), key=lambda x: -x[1]):
        print(f"  {ftype:20s} {n:4d}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mixed-form bundle PDF")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--source", type=str, default="demo_samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_mixed_bundle(args.source, args.count, args.output, args.seed)
