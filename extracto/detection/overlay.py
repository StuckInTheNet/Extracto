"""Debug overlay visualization for detected controls."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz

from extracto.detection.controls import page_to_image, process_pdf

COLOR_BOX = (0, 255, 0)
COLOR_RADIO = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_SEL = (0, 0, 255)


def draw_overlay(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    """Draw bounding boxes and labels on page images for QA review."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = process_pdf(pdf_path, dpi=dpi)
    written = []

    for i, page in enumerate(doc):
        img = page_to_image(page, dpi=dpi)
        vis = img.copy()
        page_res = results["pages"][i]

        for c in page_res["controls"]:
            x, y, w, h = c["bbox"]
            if c["kind"] == "checkbox":
                color = COLOR_SEL if c["selected"] else COLOR_BOX
            else:
                color = COLOR_SEL if c["selected"] else COLOR_RADIO
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            label = c.get("label") or ""
            if label:
                cv2.rectangle(vis, (x, max(y - 18, 0)), (x + min(240, w + 160), y), (0, 0, 0), -1)
                cv2.putText(vis, label[:28], (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        out_path = str(Path(out_dir) / f"{Path(pdf_path).stem}_page_{i + 1:03d}.png")
        cv2.imwrite(out_path, vis)
        written.append(out_path)

    return written
