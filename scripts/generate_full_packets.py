import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.graphics.barcode import code128
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF


random.seed(1337)


def _list_dataset_forms(dataset_dir: str) -> Tuple[List[Path], List[Path]]:
    med = sorted(Path(dataset_dir, "medical").glob("*.pdf"))
    ins = sorted(Path(dataset_dir, "insurance").glob("*.pdf"))
    return med, ins


def _render_page_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img


def _scan_augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # small rotation
    angle = random.uniform(-3.0, 3.0)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # slight blur
    if random.random() < 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    # add gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # random brightness/contrast
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # occasional grayscale + threshold (simulate binarized scan)
    if random.random() < 0.3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    # random horizontal streaks (fax lines)
    if random.random() < 0.4:
        for _ in range(random.randint(1, 4)):
            y = random.randint(0, h - 1)
            thickness = random.randint(1, 2)
            color = random.choice([(200, 200, 200), (230, 230, 230)])
            cv2.line(img, (0, y), (w, y), color, thickness)
    # slight perspective warp
    if random.random() < 0.3:
        dx = int(w * 0.01)
        dy = int(h * 0.01)
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[random.randint(0, dx), random.randint(0, dy)],
                          [w - random.randint(0, dx), random.randint(0, dy)],
                          [random.randint(0, dx), h - random.randint(0, dy)],
                          [w - random.randint(0, dx), h - random.randint(0, dy)]])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # edge shadow
    if random.random() < 0.5:
        mask = np.zeros((h, w), dtype=np.uint8)
        side = random.choice(['left','right','top','bottom'])
        if side in ('left','right'):
            width = random.randint(int(0.01*w), int(0.05*w))
            if side=='left':
                mask[:, :width] = 1
            else:
                mask[:, -width:] = 1
        else:
            height = random.randint(int(0.01*h), int(0.05*h))
            if side=='top':
                mask[:height, :] = 1
            else:
                mask[-height:, :] = 1
        img = cv2.addWeighted(img, 1.0, np.full_like(img, 0), 0.0, 0)
        img[mask.astype(bool)] = (img[mask.astype(bool)] * 0.9).astype(np.uint8)
    return img


def _insert_augmented_page(newdoc: fitz.Document, base_page: fitz.Page, rotate90: bool = False, header_tag: str | None = None):
    img = _render_page_image(base_page, dpi=300)
    img = _scan_augment(img)
    if rotate90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    # encode to PNG and insert
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    p = newdoc.new_page(width=w * 72.0 / 300.0, height=h * 72.0 / 300.0)
    rect = fitz.Rect(0, 0, p.rect.width, p.rect.height)
    p.insert_image(rect, stream=buf.tobytes())
    # Overlay vector header text to aid splitting without OCR
    if header_tag:
        try:
            p.insert_text(fitz.Point(24, 28), header_tag, fontsize=14, color=(0, 0, 0))
        except Exception:
            pass


def _insert_vector_page(newdoc: fitz.Document, srcdoc: fitz.Document, page_index: int = 0):
    newdoc.insert_pdf(srcdoc, from_page=page_index, to_page=page_index)


def _make_cover_sheet(path: Path, title: str, packet_id: str):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, h - 72, title)
    c.setFont("Helvetica", 12)
    c.drawString(72, h - 100, f"Packet ID: {packet_id}")
    c.drawString(72, h - 120, "This is a simulated cover sheet.")
    c.showPage()
    c.save()


def _make_separator(path: Path, label: str):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(w / 2, h - 100, "=== SEPARATOR ===")
    c.setFont("Helvetica", 12)
    c.drawCentredString(w / 2, h - 130, label)
    # barcode
    bc = code128.Code128(label, barHeight=40, humanReadable=True)
    bc.drawOn(c, (w - 300) / 2, h - 220)
    c.showPage()
    c.save()


def generate_packets(out_dir: str = "outputs/packets", dataset_dir: str = "dataset", n_packets: int = 5, min_docs: int = 5, max_docs: int = 12):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    med, ins = _list_dataset_forms(dataset_dir)
    assert med and ins, "Dataset forms not found; run scripts/generate_benchmark_docs.py first."

    manifest_all: Dict[str, Any] = {"packets": []}

    for pk in range(1, n_packets + 1):
        packet_id = f"PKT{pk:03d}"
        # Build sequence of docs
        n_docs = random.randint(min_docs, max_docs)
        parts: List[Dict[str, Any]] = []
        # First make a cover sheet
        parts.append({"kind": "cover", "label": f"Cover for {packet_id}"})
        for d in range(n_docs):
            kind = random.choice(["medical", "insurance"])
            if kind == "medical":
                src = random.choice(med)
            else:
                src = random.choice(ins)
            pages = random.randint(1, 3)
            parts.append({"kind": kind, "src": str(src), "pages": pages})
            # Occasional separator sheet
            if random.random() < 0.4 and d != n_docs - 1:
                parts.append({"kind": "separator", "label": f"{packet_id}-SEP-{d+1}"})

        # Assemble packet PDF with augmented pages
        packet_pdf = Path(out_dir) / f"packet_{packet_id}.pdf"
        newdoc = fitz.open()
        cur_page_index = 0
        truth_segments: List[Dict[str, Any]] = []
        current_seg_start = None
        current_type = None

        # Write cover
        cover_tmp = Path(out_dir) / f"{packet_id}_cover_tmp.pdf"
        _make_cover_sheet(cover_tmp, "Fax Cover Sheet", packet_id)
        cov_doc = fitz.open(str(cover_tmp))
        # Keep cover as vector so header text is machine-readable for splitting
        _insert_vector_page(newdoc, cov_doc, 0)
        cov_doc.close()
        cover_tmp.unlink(missing_ok=True)
        truth_segments.append({"type": "cover", "start": cur_page_index, "end": cur_page_index})
        cur_page_index += 1

        for p in parts[1:]:
            if p["kind"] in ("medical", "insurance"):
                src = fitz.open(p["src"])
                # Start new segment for this doc
                seg_start = cur_page_index
                pages_to_take = min(p["pages"], len(src))
                for i in range(pages_to_take):
                    base_page = src[i % len(src)]
                    rot = True if random.random() < 0.1 else False
                    # Provide a machine-readable header tag for splitter
                    tag = "Patient Intake" if p["kind"] == "medical" else "Insurance Claim"
                    _insert_augmented_page(newdoc, base_page, rotate90=rot, header_tag=tag)
                    cur_page_index += 1
                src.close()
                truth_segments.append({"type": p["kind"], "start": seg_start, "end": cur_page_index - 1})
            elif p["kind"] == "separator":
                sep_tmp = Path(out_dir) / f"{packet_id}_sep_tmp.pdf"
                _make_separator(sep_tmp, p["label"])
                sep_doc = fitz.open(str(sep_tmp))
                # Keep separator as vector so header/label text is visible for splitting
                _insert_vector_page(newdoc, sep_doc, 0)
                sep_doc.close()
                sep_tmp.unlink(missing_ok=True)
                truth_segments.append({"type": "separator", "start": cur_page_index, "end": cur_page_index})
                cur_page_index += 1

        newdoc.save(str(packet_pdf))
        newdoc.close()
        manifest_all["packets"].append({
            "packet_id": packet_id,
            "pdf": str(packet_pdf),
            "segments": truth_segments,
        })

    manifest_path = Path(out_dir) / "packets_manifest.json"
    Path(manifest_path).write_text(json.dumps(manifest_all, indent=2))
    return str(manifest_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/packets")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=12)
    args = ap.parse_args()
    mp = generate_packets(args.out, args.dataset, n_packets=args.n, min_docs=args.min_docs, max_docs=args.max_docs)
    print(mp)
