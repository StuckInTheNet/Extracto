"""Checkbox and radio button detection from PDF page images.

Three-tier detection strategy:
1. YOLO object detection (if trained model available)
2. OpenCV contour analysis with trained LR classifiers
3. Pure heuristic fallback
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from joblib import load

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# PaddleOCR disabled by default — it crashes on some systems and PyMuPDF's
# built-in Tesseract OCR is sufficient. Set EXTRACTO_USE_PADDLE=1 to enable.
PaddleOCR = None
if os.environ.get("EXTRACTO_USE_PADDLE") == "1":
    try:
        from paddleocr import PaddleOCR
    except Exception:
        PaddleOCR = None


@dataclass
class Control:
    kind: str  # 'checkbox' or 'radio'
    bbox: tuple[int, int, int, int]  # x, y, w, h in px
    selected: bool
    label: str
    conf: float = 0.0


def page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def detect_skew_angle(img: np.ndarray) -> float:
    """Detect document skew by analyzing near-horizontal edge line angles.

    Returns the median rotation angle in degrees. Positive = clockwise skew.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew_image(img: np.ndarray, min_angle: float = 0.3) -> np.ndarray:
    """Deskew an image by detecting and correcting rotation.

    Only rotates if skew exceeds min_angle degrees (avoids unnecessary work
    on clean images).
    """
    angle = detect_skew_angle(img)
    if abs(angle) < min_angle:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def make_page_from_image(img: np.ndarray, dpi: int = 300) -> fitz.Page:
    """Create a new single-page PDF document from an image, return the page.

    Used to feed a deskewed/preprocessed image through PyMuPDF's OCR pipeline
    (which operates on pages, not arbitrary images).
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    h, w = img.shape[:2]
    doc = fitz.open()
    page_w = w * 72.0 / dpi
    page_h = h * 72.0 / dpi
    page = doc.new_page(width=page_w, height=page_h)
    rect = fitz.Rect(0, 0, page_w, page_h)
    page.insert_image(rect, stream=buf.tobytes())
    return page


# --- YOLO detector ---

YOLO_MODEL_PATHS = [
    "models/yolo_selection/weights/best.pt",
    "extracto/detection/models/yolo_selection/weights/best.pt",
]
_yolo_model = None
_yolo_enabled = True  # module-level flag, toggled by process_pdf


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if YOLO is None:
        return None
    for p in YOLO_MODEL_PATHS:
        if os.path.exists(p):
            try:
                _yolo_model = YOLO(p)
                return _yolo_model
            except Exception:
                continue
    return None


# --- LR classifiers ---

_MODEL_DIR = Path(__file__).parent / "models"


def _load_lr_models() -> tuple[Any, Any]:
    cb_model, rb_model = None, None
    try:
        cb_model = load(_MODEL_DIR / "checkbox_lr.joblib")
        rb_model = load(_MODEL_DIR / "radio_lr.joblib")
    except Exception:
        pass
    return cb_model, rb_model


# --- Detection ---

def detect_controls(img: np.ndarray) -> list[Control]:
    """Detect checkboxes and radio buttons in a page image."""
    cb_model, rb_model = _load_lr_models()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    # Try YOLO first
    yolo = _get_yolo_model()
    controls: list[Control] = []
    # YOLO attempt — collect detections, will ensemble with contour/LR below.
    # Caller can disable YOLO for form types where it hurts accuracy.
    yolo_controls: list[Control] = []
    if yolo is not None and _yolo_enabled:
        try:
            res = yolo.predict(source=img, verbose=False, conf=0.30, imgsz=640, device="cpu")[0]
            for b in res.boxes:
                x0, y0, x1, y1 = map(int, b.xyxy[0].tolist())
                w, h = x1 - x0, y1 - y0
                cls = int(b.cls.item())
                score = float(b.conf.item()) if hasattr(b, "conf") else 0.5
                kind_map = {0: ("checkbox", False), 1: ("checkbox", True), 2: ("radio", False), 3: ("radio", True)}
                if cls not in kind_map:
                    continue
                kind, selected = kind_map[cls]
                yolo_controls.append(Control(kind=kind, bbox=(x0, y0, w, h), selected=selected, label="", conf=score))
            yolo_controls = _non_max_suppression(yolo_controls, iou_threshold=0.3)
        except Exception:
            yolo_controls = []

    # Fallback: contour-based detection
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 400 or area > 14000:
            continue
        # 22-95px covers most checkbox sizes. Going lower adds text-character
        # false positives that hurt scan accuracy more than they help.
        if not (22 <= w <= 95 and 22 <= h <= 95):
            continue

        ar = abs(w - h) / max(w, h)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * cv2.contourArea(cnt) / (peri * peri)

        # Interior region excluding border strokes
        border = max(2, int(min(w, h) * 0.18))
        roi = thr[y + border : y + h - border, x + border : x + w - border]
        gray_roi = gray[y + border : y + h - border, x + border : x + w - border]
        if roi.size == 0:
            continue
        erode_k = int(max(1, round(min(w, h) * 0.02)))
        roi_eroded = cv2.erode(roi, np.ones((erode_k, erode_k), np.uint8), iterations=1)
        fill_ratio = float(np.count_nonzero(roi_eroded)) / roi_eroded.size

        kind = None
        selected = False
        conf = 0.0

        # Checkbox: near-square, lower circularity
        if ar < 0.15 and circularity < 0.7:
            kind = "checkbox"
            selected, conf = _classify_checkbox(
                roi_eroded, gray_roi, fill_ratio, w, h, cb_model
            )
        # Radio: near-circle
        elif circularity >= 0.7 and 0.85 <= w / max(h, 1) <= 1.15:
            kind = "radio"
            selected, conf = _classify_radio(
                thr, x, y, w, h, fill_ratio, rb_model
            )
        else:
            continue

        controls.append(Control(kind=kind, bbox=(x, y, w, h), selected=selected, label="", conf=conf))

    contour_controls = _non_max_suppression(controls, iou_threshold=0.3)

    # ENSEMBLE: combine YOLO + contour detections.
    # NOTE: LR re-classification of YOLO detections was tested but hurts accuracy
    # because the LR models were trained on clean-image features, not YOLO-crop
    # features. Disabled until LR is retrained on YOLO-sourced crops.
    if False and yolo_controls and cb_model is not None:
        for yc in yolo_controls:
            x, y, w, h = yc.bbox
            # Extract features from the image at the YOLO-detected bbox
            border = max(2, int(min(w, h) * 0.18))
            roi = thr[y + border : y + h - border, x + border : x + w - border]
            gray_roi = gray[y + border : y + h - border, x + border : x + w - border]
            if roi.size == 0 or gray_roi.size == 0:
                continue
            erode_k = int(max(1, round(min(w, h) * 0.02)))
            roi_eroded = cv2.erode(roi, np.ones((erode_k, erode_k), np.uint8), iterations=1)
            fill_ratio = float(np.count_nonzero(roi_eroded)) / max(1, roi_eroded.size)
            edges = cv2.Canny(gray_roi, 80, 200)
            edge_ratio = float(np.count_nonzero(edges)) / max(1, edges.size)

            ar = abs(w - h) / max(w, h)
            peri = cv2.arcLength(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]), True)
            circularity = 4 * np.pi * w * h / (peri * peri) if peri > 0 else 0

            if yc.kind == "checkbox" or circularity < 0.7:
                # Re-classify using checkbox LR
                lines_det = cv2.HoughLinesP(cv2.Canny(roi_eroded, 80, 200), 1, np.pi/180, threshold=30, minLineLength=int(min(w,h)*0.55), maxLineGap=2)
                line_cnt = 0 if lines_det is None else min(20, len(lines_det))
                chk = _check_template_score(edges, w, h)
                feats = np.array([[fill_ratio, edge_ratio, float(line_cnt), chk, 0.0, 0.0, float(w), float(h)]])
                sel_prob = cb_model.predict_proba(feats)[0, 1]
                yc.selected = bool(sel_prob >= 0.5)
                yc.conf = float(sel_prob)
            elif rb_model is not None:
                # Re-classify using radio LR
                r_in = max(1, int(min(w, h) * 0.18))
                r_out = max(r_in + 1, int(min(w, h) * 0.45))
                H, W = roi.shape[:2]
                yy2, xx2 = np.ogrid[-H//2:H//2, -W//2:W//2]
                dist = np.sqrt(xx2**2 + yy2**2)
                center_m = dist <= r_in
                annulus_m = (dist > r_in) & (dist <= r_out)
                cr = float(np.count_nonzero(roi[center_m])) / max(1, int(center_m.sum()))
                ar_val = float(np.count_nonzero(roi[annulus_m])) / max(1, int(annulus_m.sum()))
                feats = np.array([[fill_ratio, 0.0, 0.0, 0.0, cr, ar_val, float(w), float(h)]])
                sel_prob = rb_model.predict_proba(feats)[0, 1]
                yc.selected = bool(sel_prob >= 0.5)
                yc.conf = float(sel_prob)

    if not yolo_controls:
        return contour_controls
    if not contour_controls:
        return yolo_controls

    # Merge: add contour detections that don't overlap YOLO detections
    combined = list(yolo_controls)
    for cc in contour_controls:
        overlap = False
        for yc in yolo_controls:
            if _iou(cc.bbox, yc.bbox) > 0.4:
                overlap = True
                break
        if not overlap:
            combined.append(cc)
    return _non_max_suppression(combined, iou_threshold=0.3)


def _classify_checkbox(
    roi_eroded: np.ndarray,
    gray_roi: np.ndarray,
    fill_ratio: float,
    w: int,
    h: int,
    model: Any,
) -> tuple[bool, float]:
    """Determine if a checkbox is selected using lines, template matching, and optional ML."""
    # Detect X marks via line segments
    thin = cv2.ximgproc.thinning(roi_eroded) if hasattr(cv2, "ximgproc") else cv2.Canny(roi_eroded, 80, 200)
    lines = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=30, minLineLength=int(min(w, h) * 0.55), maxLineGap=2)
    line_angles = []
    if lines is not None:
        for l in lines[:20]:
            x1, y1, x2, y2 = l[0]
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            line_angles.append(ang)

    has_cross = _detect_cross(line_angles)

    edges = cv2.Canny(gray_roi, 80, 200)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size if edges.size else 0.0
    chk_score = _check_template_score(edges, w, h)

    if model is not None:
        feats = np.array([[fill_ratio, edge_ratio, float(len(line_angles)), chk_score, 0.0, 0.0, float(w), float(h)]])
        sel_prob = model.predict_proba(feats)[0, 1]
        return bool(sel_prob >= 0.5), float(sel_prob)

    selected = fill_ratio > 0.22 or has_cross or edge_ratio > 0.02 or chk_score > 0.35
    conf = min(0.95, max(fill_ratio, edge_ratio * 3, chk_score))
    return selected, conf


def _classify_radio(
    thr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fill_ratio: float,
    model: Any,
) -> tuple[bool, float]:
    """Determine if a radio button is selected using radial fill analysis and optional ML."""
    r = int(min(w, h) * 0.3)
    cx, cy = x + w // 2, y + h // 2
    y0, y1 = max(cy - r, 0), min(cy + r, thr.shape[0])
    x0, x1 = max(cx - r, 0), min(cx + r, thr.shape[1])
    inner = thr[y0:y1, x0:x1]

    if inner.size == 0:
        return False, 0.0

    inner_fill = float(np.count_nonzero(inner)) / inner.size
    r_in = max(1, int(min(w, h) * 0.18))
    r_out = max(r_in + 1, int(min(w, h) * 0.45))
    yy, xx = np.ogrid[-inner.shape[0] // 2 : inner.shape[0] // 2, -inner.shape[1] // 2 : inner.shape[1] // 2]
    dist = np.sqrt(xx**2 + yy**2)
    center_mask = dist <= r_in
    annulus_mask = (dist > r_in) & (dist <= r_out)
    center_ratio = float(np.count_nonzero(inner[center_mask])) / max(1, int(center_mask.sum()))
    annulus_ratio = float(np.count_nonzero(inner[annulus_mask])) / max(1, int(annulus_mask.sum()))

    if model is not None:
        feats = np.array([[fill_ratio, 0.0, 0.0, 0.0, center_ratio, annulus_ratio, float(w), float(h)]])
        sel_prob = model.predict_proba(feats)[0, 1]
        return bool(sel_prob >= 0.5), float(sel_prob)

    selected = inner_fill > 0.25 or (center_ratio > 0.35 and center_ratio > annulus_ratio * 1.5)
    conf = float(max(inner_fill, center_ratio))
    return selected, conf


def _detect_cross(line_angles: list[float]) -> bool:
    """Check if line angles suggest an X pattern."""
    if len(line_angles) < 2:
        return False
    for a in line_angles:
        for b in line_angles:
            d = abs(a - b)
            if 70 <= d <= 110 and ((30 <= a <= 70) or (110 <= a <= 150)) and ((30 <= b <= 70) or (110 <= b <= 150)):
                return True
    return False


def _check_template_score(edges: np.ndarray, w: int, h: int) -> float:
    """Score edges against a procedural checkmark template."""
    best = 0.0
    base = max(10, min(28, int(min(w, h) * 0.6)))
    for scale in (0.8, 1.0, 1.2):
        sz = max(8, int(base * scale))
        tpl = np.zeros((sz, sz), dtype=np.uint8)
        cv2.line(tpl, (1, sz // 2), (sz // 3, sz - 2), 255, 1)
        cv2.line(tpl, (sz // 3, sz - 2), (sz - 2, 2), 255, 1)
        if edges.shape[0] < tpl.shape[0] or edges.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(edges, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, _ = cv2.minMaxLoc(res)
        best = max(best, float(maxv))
    return best


# --- NMS ---

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _non_max_suppression(controls: list[Control], iou_threshold: float = 0.3) -> list[Control]:
    if not controls:
        return []
    boxes = np.array([c.bbox for c in controls])
    scores = np.array([boxes[:, 2] * boxes[:, 3]])
    idxs = scores.argsort(axis=1)[0][::-1]
    picked = []
    used = set()
    for i in idxs:
        if i in used:
            continue
        keep = True
        for j in picked:
            if _iou(controls[i].bbox, controls[j].bbox) > iou_threshold:
                keep = False
                break
        if keep:
            picked.append(i)
        used.add(i)
    return [controls[i] for i in picked]


# --- OCR fallback ---

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


# --- Text extraction ---

def extract_lines_from_words(page: fitz.Page) -> list[tuple[str, tuple[float, float, float, float], tuple[int, int]]]:
    words = page.get_text("words")
    lines: defaultdict[tuple[int, int], list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for x0, y0, x1, y1, w, bno, lno, _ in words:
        lines[(int(bno), int(lno))].append((w, (x0, y0, x1, y1)))
    results = []
    for key, items in lines.items():
        items.sort(key=lambda t: t[1][0])
        text = " ".join(w for w, _ in items).strip()
        x0 = min(b[0] for _, b in items)
        y0 = min(b[1] for _, b in items)
        x1 = max(b[2] for _, b in items)
        y1 = max(b[3] for _, b in items)
        results.append((text, (x0, y0, x1, y1), key))
    return results


def _ocr_via_pymupdf(
    page: fitz.Page,
) -> list[tuple[str, tuple[float, float, float, float], tuple[int, int]]]:
    """OCR a page using PyMuPDF's built-in Tesseract integration.

    This is the preferred OCR path because it requires no additional Python
    packages — just the system Tesseract binary (pre-installed on most Macs
    and available via apt/brew on Linux).

    Returns text lines with bboxes in the same format as extract_lines_from_words.
    """
    try:
        tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
        words = page.get_text("words", textpage=tp)
    except Exception:
        return []

    if not words:
        return []

    # OCR word correction is opt-in — too aggressive, can corrupt valid words.
    # Fuzzy matching at the structuring layer handles OCR errors more safely.
    if os.environ.get("EXTRACTO_OCR_CORRECT") == "1":
        try:
            from extracto.structuring.fuzzy import correct_word
            words = [
                (x0, y0, x1, y1, correct_word(w), bno, lno, wn)
                for x0, y0, x1, y1, w, bno, lno, wn in words
            ]
        except Exception:
            pass

    # Group OCR'd words into lines using the same logic as extract_lines_from_words
    from collections import defaultdict

    lines: defaultdict[tuple[int, int], list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for x0, y0, x1, y1, w, bno, lno, _ in words:
        lines[(int(bno), int(lno))].append((w, (x0, y0, x1, y1)))

    results = []
    for key, items in lines.items():
        items.sort(key=lambda t: t[1][0])
        text = " ".join(w for w, _ in items).strip()
        x0 = min(b[0] for _, b in items)
        y0 = min(b[1] for _, b in items)
        x1 = max(b[2] for _, b in items)
        y1 = max(b[3] for _, b in items)
        if text:
            results.append((text, (x0, y0, x1, y1), key))
    return results


def extract_lines(
    page: fitz.Page, img: np.ndarray | None = None
) -> list[tuple[str, tuple[float, float, float, float], tuple[int, int]]]:
    """Extract text lines from a page.

    Three-tier fallback:
    1. PDF-native text (get_text "words") — fastest, works on digital PDFs
    2. PyMuPDF built-in OCR (Tesseract) — for scanned PDFs, no extra packages
    3. PaddleOCR — optional, for when Tesseract isn't available
    """
    # Tier 1: PDF-native text
    try:
        results = extract_lines_from_words(page)
        if results:
            return results, "native"
    except Exception:
        pass

    # Tier 2: PyMuPDF built-in OCR (uses system Tesseract)
    try:
        ocr_results = _ocr_via_pymupdf(page)
        if ocr_results:
            return ocr_results, "ocr"
    except Exception:
        pass

    # Tier 3: PaddleOCR (optional dependency)
    if img is None:
        try:
            img = page_to_image(page, dpi=300)
        except Exception:
            return []

    ocr = _get_paddle_ocr()
    if ocr is None or img is None:
        return [], "none"

    try:
        ocr_res = ocr.ocr(img, cls=True)
    except Exception:
        return [], "none"

    lines_out = []
    for block in ocr_res:
        for line in block:
            poly = line[0]
            text = line[1][0]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            lines_out.append((text, (x0, y0, x1, y1), (0, 0)))
    return lines_out, "ocr"


# --- Label association ---

def label_controls(
    controls: list[Control],
    lines: list[tuple[str, tuple[float, float, float, float], tuple[int, int]]],
    scale: float,
) -> None:
    """Associate each control with its nearest text label."""
    for c in controls:
        x, y, w, h = c.bbox
        cx, cy = (x + w / 2) / scale, (y + h / 2) / scale
        right_candidates = []
        nearest_any = (None, 1e9)
        for text, (bx, by, ex, ey), _ in lines:
            ly = (by + ey) / 2
            if abs(ly - cy) > 12:
                continue
            dx_right = bx - cx
            if 0 <= dx_right <= 80:
                right_candidates.append((dx_right, text))
            center_x = (bx + ex) / 2
            dcenter = abs(center_x - cx)
            if dcenter < nearest_any[1]:
                nearest_any = (text, dcenter)
        if right_candidates:
            right_candidates.sort(key=lambda t: t[0])
            c.label = right_candidates[0][1].strip()
        else:
            c.label = (nearest_any[0] or "").strip()


# --- Main entry point ---

def process_pdf(path: str, dpi: int = 300, use_yolo: bool = True) -> dict[str, Any]:
    """Extract controls and text lines from all pages of a PDF.

    For digitally-generated PDFs, uses PDF-native vector drawing extraction
    (far more accurate than CV). Falls back to CV detection for scanned PDFs.

    Args:
        use_yolo: Whether to use YOLO detection on scanned pages. Set False
            for form types where YOLO hurts accuracy (medical, insurance).
    """
    from extracto.detection.pdf_native import extract_controls_from_drawings, has_useful_drawings

    global _yolo_enabled
    _yolo_enabled = use_yolo

    doc = fitz.open(path)
    results: dict[str, Any] = {"file": path, "pages": []}
    scale = dpi / 72.0

    for page in doc:
        # Try PDF-native extraction first
        native_controls = []
        if has_useful_drawings(page):
            native_controls = extract_controls_from_drawings(page, scale=scale)

        deskewed_page = None  # Set when we deskew a scanned page

        if native_controls:
            controls = [
                Control(
                    kind=c["kind"],
                    bbox=c["bbox"],
                    selected=c["selected"],
                    label="",
                    conf=c["conf"],
                )
                for c in native_controls
            ]
            controls = _non_max_suppression(controls, iou_threshold=0.3)
        else:
            # Scanned/raster page: deskew before detection for better CV + OCR accuracy
            img = page_to_image(page, dpi=dpi)
            deskewed = deskew_image(img, min_angle=0.3)
            controls = detect_controls(deskewed)

            # MULTI-DPI ENSEMBLE: disabled — causes memory crashes when combined
            # with YOLO model loading + large page rendering.
            # TODO: re-enable when memory usage is optimized.
            if False and len(controls) < 8:
                try:
                    hi_dpi = 450
                    hi_img = page_to_image(page, dpi=hi_dpi)
                    hi_deskewed = deskew_image(hi_img, min_angle=0.3)
                    hi_controls = detect_controls(hi_deskewed)
                    ratio = dpi / hi_dpi
                    rescaled = []
                    for c in hi_controls:
                        x, y, w, h = c.bbox
                        rescaled.append(Control(
                            kind=c.kind,
                            bbox=(int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)),
                            selected=c.selected,
                            label="",
                            conf=c.conf,
                        ))
                    if rescaled:
                        combined = list(controls)
                        for hc in rescaled:
                            overlap = False
                            for pc in controls:
                                if _iou(hc.bbox, pc.bbox) > 0.3:
                                    overlap = True
                                    break
                            if not overlap:
                                combined.append(hc)
                        controls = _non_max_suppression(combined, iou_threshold=0.3)
                except Exception:
                    pass

        # OCR: use the original page (Tesseract handles slight rotation fine)
        # Avoids re-rendering deskewed images which wastes memory.
        lines, text_source = extract_lines(page, img=None)
        label_controls(controls, lines, scale)

        results["pages"].append({
            "controls": [
                {"kind": c.kind, "bbox": c.bbox, "selected": c.selected, "label": c.label, "conf": c.conf}
                for c in controls
            ],
            "lines": [{"text": t, "bbox": list(b)} for t, b, _ in lines],
            "size": [float(page.rect.width), float(page.rect.height)],
            "text_source": text_source,
            "detection_mode": "native" if native_controls else "cv",
        })

    return results
