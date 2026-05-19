import os
import json
import random
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def bundle_mixed(manifest_path: str, out_path: str):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pdfs = ([entry["pdf"] for entry in manifest.get("medical", [])] +
            [entry["pdf"] for entry in manifest.get("insurance", [])])

    random.seed(123)
    random.shuffle(pdfs)

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    for p in pdfs:
        reader = PdfReader(p)
        for page in reader.pages:
            writer.add_page(page)
    with open(out_path, "wb") as f:
        writer.write(f)
    return pdfs


if __name__ == "__main__":
    manifest = os.environ.get("MANIFEST", "dataset/manifest.json")
    out_path = os.environ.get("OUT", "dataset/mixed/mixed.pdf")
    used = bundle_mixed(manifest, out_path)
    print(f"Wrote {out_path} with {len(used)} documents")
