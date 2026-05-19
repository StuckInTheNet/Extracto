"""Generate scanned/degraded versions of clean synthetic forms.

Takes clean digital PDFs and applies realistic scan degradation:
- Rotation (±2 degrees)
- Gaussian blur
- Gaussian noise
- Random brightness/contrast shifts
- Fax line artifacts
- Slight perspective warp
- Binarization (simulating poor photocopy)

The output PDFs contain ONLY raster images — no vector drawings.
This forces the CV fallback path and OCR text extraction, testing
the pipeline under real-world conditions.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

random.seed(5000)
np.random.seed(5000)


def page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def scan_augment(img: np.ndarray, severity: str = "medium") -> np.ndarray:
    """Apply realistic scan degradation to an image.

    Severity levels:
    - light: minor rotation + slight blur (good scanner)
    - medium: rotation + blur + noise + brightness shift (typical office scan)
    - heavy: all of the above + fax lines + perspective warp + binarization (bad fax)
    """
    h, w = img.shape[:2]

    # Rotation
    if severity == "light":
        angle = random.uniform(-0.5, 0.5)
    elif severity == "medium":
        angle = random.uniform(-1.5, 1.5)
    else:
        angle = random.uniform(-3.0, 3.0)

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Gaussian blur
    if severity in ("medium", "heavy"):
        ksize = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Gaussian noise
    if severity in ("medium", "heavy"):
        noise_std = 6 if severity == "medium" else 12
        noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Brightness / contrast shift
    if severity != "light":
        alpha = random.uniform(0.92, 1.08)
        beta = random.uniform(-8, 8)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Fax lines (horizontal streaks)
    if severity == "heavy" and random.random() < 0.5:
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, h - 1)
            thickness = random.randint(1, 2)
            color = random.choice([(200, 200, 200), (220, 220, 220)])
            cv2.line(img, (0, y), (w, y), color, thickness)

    # Perspective warp
    if severity == "heavy" and random.random() < 0.3:
        dx = int(w * 0.008)
        dy = int(h * 0.008)
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Binarization (poor photocopy)
    if severity == "heavy" and random.random() < 0.2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    return img


def degrade_pdf(input_pdf: str, output_pdf: str, severity: str = "medium", dpi: int = 300):
    """Convert a clean PDF to a scanned/degraded version.

    The output PDF contains only raster images — all vector drawings are gone.
    """
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()

    for page in doc:
        img = page_to_image(page, dpi=dpi)
        img = scan_augment(img, severity=severity)

        # Encode as PNG
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue

        # Create new page with the degraded image
        img_h, img_w = img.shape[:2]
        page_w = img_w * 72.0 / dpi
        page_h = img_h * 72.0 / dpi
        new_page = new_doc.new_page(width=page_w, height=page_h)
        rect = fitz.Rect(0, 0, page_w, page_h)
        new_page.insert_image(rect, stream=buf.tobytes())

    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()


def generate_scanned_dataset(
    clean_dir: str,
    out_dir: str,
    severity: str = "medium",
    max_forms: int = 10,
    form_type: str = "",
):
    """Degrade a directory of clean PDFs into scanned versions.

    Copies the ground truth JSON unchanged — same expected output,
    harder input.
    """
    clean_path = Path(clean_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(clean_path.glob("*.pdf"))[:max_forms]
    manifest_key = form_type or clean_path.name

    manifest = {manifest_key: []}
    for pdf in pdfs:
        json_file = pdf.with_suffix(".json")
        if not json_file.exists():
            continue

        out_pdf = out_path / pdf.name
        out_json = out_path / json_file.name

        print(f"  Degrading {pdf.name} ({severity})...")
        degrade_pdf(str(pdf), str(out_pdf), severity=severity)

        # Copy ground truth unchanged
        shutil.copy2(str(json_file), str(out_json))

        manifest[manifest_key].append({
            "pdf": str(out_pdf),
            "json": str(out_json),
        })

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    severity = os.environ.get("SEVERITY", "medium")
    n = int(os.environ.get("N_FORMS", "10"))

    form_dirs = [
        ("dataset/medical", "medical"),
        ("dataset/insurance", "insurance"),
        ("dataset/cms1500", "cms1500"),
        ("dataset/phq9", "phq9"),
        ("dataset/hipaa", "hipaa"),
        ("dataset/froi", "froi"),
        ("dataset/eob", "eob"),
    ]

    for clean_dir, form_type in form_dirs:
        if not Path(clean_dir).exists():
            continue
        out_dir = f"dataset/scanned_{severity}/{form_type}"
        print(f"\n=== {form_type} ({severity}) ===")
        generate_scanned_dataset(clean_dir, out_dir, severity=severity,
                                 max_forms=n, form_type=form_type)

    print(f"\nDone. Scanned forms in dataset/scanned_{severity}/")


if __name__ == "__main__":
    main()
