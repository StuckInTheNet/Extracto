"""Table extraction using Microsoft TableTransformer + cell-level OCR.

For scanned documents where dense financial tables are illegible to
standard OCR, this module:
1. Detects tables via TableTransformer object detection
2. Recognizes table structure (rows + columns)
3. Crops each cell and OCRs it individually
4. Returns structured cell data

Requires: transformers, timm, torch
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import fitz
import numpy as np

_models_loaded = False
_detection_model = None
_detection_processor = None
_structure_model = None
_structure_processor = None


def _load_models():
    global _models_loaded, _detection_model, _detection_processor
    global _structure_model, _structure_processor
    if _models_loaded:
        return True
    try:
        import torch
        import gc

        # Free YOLO model memory before loading TableTransformer
        from extracto.detection.controls import _yolo_model
        if _yolo_model is not None:
            import extracto.detection.controls as _ctrl
            _ctrl._yolo_model = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        _detection_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        _detection_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        _structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _models_loaded = True
        return True
    except Exception:
        return False


def _ocr_cell(cell_img: np.ndarray, dpi: int = 300) -> str:
    """OCR a single table cell image via PyMuPDF Tesseract."""
    if cell_img.size == 0:
        return ""
    ok, buf = cv2.imencode(".png", cell_img)
    if not ok:
        return ""
    try:
        doc = fitz.open()
        h, w = cell_img.shape[:2]
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        page.insert_image(fitz.Rect(0, 0, w * 72 / dpi, h * 72 / dpi), stream=buf.tobytes())
        tp = page.get_textpage_ocr(language="eng", dpi=dpi, full=True)
        text = page.get_text(textpage=tp).strip()
        doc.close()
        return text
    except Exception:
        return ""


def extract_table_cells_subprocess(pdf_path: str) -> list[list[list[str]]]:
    """Extract table cells by running TableTransformer in a SUBPROCESS.

    This avoids memory conflicts with YOLO — both models can't coexist
    in the same process on machines with limited memory.
    """
    import subprocess
    import sys

    # Self-contained script — does NOT import extracto to avoid loading YOLO
    script = f'''
import json, sys, os
import fitz, numpy as np, cv2, torch, re
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

proc = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
sproc = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
smodel = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

doc = fitz.open("{pdf_path}")
page = doc[0]
mat = fitz.Matrix(300/72, 300/72)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
pil = Image.fromarray(img)

inputs = proc(images=pil, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
sizes = torch.tensor([pil.size[::-1]])
det = proc.post_process_object_detection(outputs, threshold=0.7, target_sizes=sizes)[0]

tables = []
for score, label, box in zip(det["scores"], det["labels"], det["boxes"]):
    x0, y0, x1, y1 = [int(b) for b in box.tolist()]
    crop = pil.crop((x0, y0, x1, y1))
    crop_np = np.array(crop)
    si = sproc(images=crop, return_tensors="pt")
    with torch.no_grad():
        so = smodel(**si)
    ss = torch.tensor([crop.size[::-1]])
    sr = sproc.post_process_object_detection(so, threshold=0.5, target_sizes=ss)[0]
    rows_b = sorted([list(map(int, b.tolist())) for s, l, b in zip(sr["scores"], sr["labels"], sr["boxes"]) if smodel.config.id2label[l.item()] == "table row" and s > 0.5], key=lambda b: b[1])
    cols_b = sorted([list(map(int, b.tolist())) for s, l, b in zip(sr["scores"], sr["labels"], sr["boxes"]) if smodel.config.id2label[l.item()] == "table column" and s > 0.5], key=lambda b: b[0])
    if not rows_b or not cols_b: continue
    parsed = []
    for rx0, ry0, rx1, ry1 in rows_b:
        row_cells = []
        for cx0, cy0, cx1, cy1 in cols_b:
            cx0m, cy0m = max(cx0, rx0), max(cy0, ry0)
            cx1m, cy1m = min(cx1, rx1), min(cy1, ry1)
            if cx1m <= cx0m or cy1m <= cy0m:
                row_cells.append("")
                continue
            cell = crop_np[cy0m:cy1m, cx0m:cx1m]
            if cell.size == 0:
                row_cells.append("")
                continue
            ok, buf = cv2.imencode(".png", cell)
            if not ok:
                row_cells.append("")
                continue
            cd = fitz.open()
            h, w = cell.shape[:2]
            cp = cd.new_page(width=w*72/300, height=h*72/300)
            cp.insert_image(fitz.Rect(0, 0, w*72/300, h*72/300), stream=buf.tobytes())
            tp = cp.get_textpage_ocr(language="eng", dpi=300, full=True)
            ct = " ".join(cp.get_text(textpage=tp).split())
            row_cells.append(ct)
            cd.close()
        parsed.append(row_cells)
    tables.append(parsed)
doc.close()
print(json.dumps(tables))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
            cwd="/Users/mattfish/Extracto",
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return []


def extract_table_cells(
    pdf_path: str,
    dpi: int = 300,
    detection_threshold: float = 0.7,
    structure_threshold: float = 0.5,
) -> list[list[list[str]]]:
    """Extract all tables from a PDF page as lists of cell text.

    Returns: list of tables, each table is list of rows, each row is list of cell strings.
    """
    if not _load_models():
        return []

    import torch
    from PIL import Image

    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    pil_img = Image.fromarray(img)

    # Step 1: detect tables
    inputs = _detection_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = _detection_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]])
    det_results = _detection_processor.post_process_object_detection(
        outputs, threshold=detection_threshold, target_sizes=target_sizes
    )[0]

    tables: list[list[list[str]]] = []

    for score, label, box in zip(det_results["scores"], det_results["labels"], det_results["boxes"]):
        x0, y0, x1, y1 = [int(b) for b in box.tolist()]
        table_crop = pil_img.crop((x0, y0, x1, y1))
        table_np = np.array(table_crop)

        # Step 2: recognize structure
        s_inputs = _structure_processor(images=table_crop, return_tensors="pt")
        with torch.no_grad():
            s_outputs = _structure_model(**s_inputs)
        s_sizes = torch.tensor([table_crop.size[::-1]])
        s_results = _structure_processor.post_process_object_detection(
            s_outputs, threshold=structure_threshold, target_sizes=s_sizes
        )[0]

        row_boxes = []
        col_boxes = []
        for s_score, s_label, s_box in zip(s_results["scores"], s_results["labels"], s_results["boxes"]):
            name = _structure_model.config.id2label[s_label.item()]
            sb = [int(b) for b in s_box.tolist()]
            if name == "table row" and s_score > structure_threshold:
                row_boxes.append(sb)
            elif name == "table column" and s_score > structure_threshold:
                col_boxes.append(sb)

        row_boxes.sort(key=lambda b: b[1])
        col_boxes.sort(key=lambda b: b[0])

        if not row_boxes or not col_boxes:
            continue

        # Step 3: extract cells
        parsed_rows: list[list[str]] = []
        for rx0, ry0, rx1, ry1 in row_boxes:
            row_cells: list[str] = []
            for cx0, cy0, cx1, cy1 in col_boxes:
                cell_x0 = max(cx0, rx0)
                cell_y0 = max(cy0, ry0)
                cell_x1 = min(cx1, rx1)
                cell_y1 = min(cy1, ry1)
                if cell_x1 <= cell_x0 or cell_y1 <= cell_y0:
                    row_cells.append("")
                    continue
                cell = table_np[cell_y0:cell_y1, cell_x0:cell_x1]
                cell_text = _ocr_cell(cell, dpi=dpi)
                # Clean multiline OCR artifacts
                cell_text = " ".join(cell_text.split())
                row_cells.append(cell_text)
            parsed_rows.append(row_cells)

        tables.append(parsed_rows)

    doc.close()
    return tables


def parse_eob_table(tables: list[list[list[str]]]) -> dict[str, Any]:
    """Parse EOB-specific fields from extracted table cells.

    Looks for CPT codes, dollar values, and reason codes in the cell data.
    """
    result: dict[str, Any] = {}
    cpts: list[str] = []
    total_billed = 0.0
    total_allowed = 0.0
    total_plan_paid = 0.0
    reason_codes: set[str] = set()

    CPT_RE = re.compile(r"\b(\d{5})\b")
    DOLLAR_RE = re.compile(r"\$?([\d,]+\.\d{2})")
    REASON_RE = re.compile(r"\b([A-Z]{2}-\d{1,4})\b")

    for table in tables:
        for row_idx, row in enumerate(table):
            if row_idx == 0:
                continue  # skip header row

            # Check if this is a TOTALS row
            is_totals = any("total" in cell.lower() for cell in row[:2])

            for cell_idx, cell in enumerate(row):
                # CPT codes (column 1-2 typically)
                if cell_idx <= 2 and not is_totals:
                    for m in CPT_RE.finditer(cell):
                        code = m.group(1)
                        if 10000 <= int(code) <= 99999 and code not in cpts:
                            cpts.append(code)

                # Dollar values
                for m in DOLLAR_RE.finditer(cell):
                    try:
                        val = float(m.group(1).replace(",", ""))
                        if is_totals:
                            # Assign by column position
                            if cell_idx == 2:
                                total_billed = val
                            elif cell_idx == 3:
                                total_allowed = val
                            elif cell_idx == 7:
                                total_plan_paid = val
                    except ValueError:
                        pass

                # Reason codes
                for m in REASON_RE.finditer(cell):
                    reason_codes.add(m.group(1))

    if cpts:
        result["service_line_cpts"] = sorted(cpts)
        result["service_line_count"] = len(cpts)
    if total_billed > 0:
        result["total_billed"] = total_billed
    if total_allowed > 0:
        result["total_allowed"] = total_allowed
    if total_plan_paid > 0:
        result["total_plan_paid"] = total_plan_paid
    if reason_codes:
        result["reason_codes_from_table"] = sorted(reason_codes)

    return result
