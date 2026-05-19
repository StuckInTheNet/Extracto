"""PDF splitting: segments multi-form packets into individual documents.

Uses header-based heuristics with Jaccard similarity and hysteresis
to avoid false splits on minor header variations.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import fitz

PaddleOCR = None
if os.environ.get("EXTRACTO_USE_PADDLE") == "1":
    try:
        from paddleocr import PaddleOCR
    except Exception:
        PaddleOCR = None

import numpy as np

_paddle_ocr = None


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    if PaddleOCR is None:
        return None
    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        det_dir = os.environ.get("PADDLE_OCR_DET")
        rec_dir = os.environ.get("PADDLE_OCR_REC")
        cls_dir = os.environ.get("PADDLE_OCR_CLS")
        init_kwargs = {"use_textline_orientation": True, "lang": "en"}
        if det_dir and rec_dir:
            init_kwargs.update({"det_model_dir": det_dir, "rec_model_dir": rec_dir})
            if cls_dir:
                init_kwargs["cls_model_dir"] = cls_dir
        _paddle_ocr = PaddleOCR(**init_kwargs)
        return _paddle_ocr
    except Exception:
        return None


HEADER_HEIGHT_PT = 1.5 * 72  # top 1.5 inches

ANCHORS = [
    (re.compile(r"patient intake|intake form", re.I), "medical_intake"),
    (re.compile(r"insurance claim|claim form", re.I), "insurance_claim"),
    (re.compile(r"authorization|hipaa|release", re.I), "authorization"),
    (re.compile(r"fax cover|cover sheet", re.I), "cover"),
    (re.compile(r"separator|===\s*separator\s*===", re.I), "separator"),
]


def page_header_tokens(page: fitz.Page) -> list[str]:
    """Extract normalized tokens from the top header region of a page."""
    words = page.get_text("words")
    toks = []
    for x0, y0, x1, y1, w, *_ in words:
        if y0 <= HEADER_HEIGHT_PT:
            t = re.sub(r"[^A-Za-z0-9]+", " ", w).strip().lower()
            if t:
                toks.append(t)
    if toks:
        return toks

    # Fallback: OCR the header region
    ocr = _get_paddle_ocr()
    if ocr is None:
        return []
    try:
        rect = fitz.Rect(0, 0, page.rect.width, HEADER_HEIGHT_PT)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=rect, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        res = ocr.ocr(img, cls=True)
        out = []
        for block in res:
            for line in block:
                t = line[1][0]
                t = re.sub(r"[^A-Za-z0-9]+", " ", t).strip().lower()
                if t:
                    out.extend(t.split())
        return out
    except Exception:
        return []


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _guess_type(tokens: list[str]) -> str | None:
    header = " ".join(tokens)
    for rx, label in ANCHORS:
        if rx.search(header):
            return label
    return None


def split_pdf(
    input_pdf: str,
    out_dir: str,
    sim_threshold: float = 0.2,
    hysteresis: int = 1,
) -> dict[str, Any]:
    """Split a multi-form PDF into individual document segments.

    Two-pass approach:
    1. Collect header tokens and classify each page
    2. Segment using similarity with hysteresis to avoid false splits
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(input_pdf)

    pages = []
    for i, page in enumerate(doc):
        toks = page_header_tokens(page)
        pages.append({"idx": i, "tokens": set(toks), "type": _guess_type(toks)})

    segments: list[dict[str, Any]] = []
    if not pages:
        manifest = {"source": input_pdf, "segments": []}
        Path(out_dir, f"{Path(input_pdf).stem}_split_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    cur_start = 0
    cur_type = pages[0]["type"]
    low_sim_run = 0

    def flush_segment(end_idx: int, seg_type: str | None):
        nonlocal cur_start
        segments.append({"start": cur_start, "end": end_idx, "type": seg_type})
        cur_start = end_idx + 1

    for i in range(1, len(pages)):
        prev = pages[i - 1]
        curr = pages[i]

        if curr["type"] in ("separator", "cover"):
            flush_segment(i - 1, cur_type)
            flush_segment(i, curr["type"])
            cur_type = None
            low_sim_run = 0
            continue

        if curr["type"] and curr["type"] != cur_type:
            flush_segment(i - 1, cur_type)
            cur_type = curr["type"]
            low_sim_run = 0
            continue

        sim = _jaccard(prev["tokens"], curr["tokens"])
        if sim < sim_threshold:
            low_sim_run += 1
        else:
            low_sim_run = 0

        if low_sim_run >= hysteresis:
            flush_segment(i - hysteresis, cur_type)
            cur_type = curr["type"]
            low_sim_run = 0

    if cur_start <= len(pages) - 1:
        flush_segment(len(pages) - 1, cur_type)

    # Write segment PDFs
    written = []
    base = Path(input_pdf).stem
    for idx, seg in enumerate(segments, start=1):
        start, end = seg["start"], seg["end"]
        if end < start:
            continue
        out_path = Path(out_dir) / f"{base}_part_{idx:03d}.pdf"
        new = fitz.open()
        for pno in range(start, end + 1):
            new.insert_pdf(doc, from_page=pno, to_page=pno)
        new.save(str(out_path))
        new.close()
        written.append({"pdf": str(out_path), "start": start, "end": end, "type": seg.get("type")})

    manifest = {"source": input_pdf, "segments": written}
    manifest_path = Path(out_dir) / f"{base}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
