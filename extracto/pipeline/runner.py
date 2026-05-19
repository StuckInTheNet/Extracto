"""Full pipeline orchestration: split -> detect -> structure -> report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extracto.detection.controls import process_pdf
from extracto.postprocess.llm import normalize_output
from extracto.splitting.splitter import split_pdf
from extracto.structuring.forms import structure_page


def field_conf_from_controls(page: dict[str, Any], field: str, default: float = 0.0) -> float:
    """Extract confidence for a specific field from raw control detections."""
    ctrls = page.get("controls", [])

    if field == "sex":
        labels = {"Male", "Female", "Other"}
        confs = [c.get("conf", 0.0) for c in ctrls if c.get("label") in labels and c.get("selected")]
        return max(confs) if confs else default

    if field in ("Smoker", "Diabetic", "Is this work-related?", "Auto accident?"):
        confs = [c.get("conf", 0.0) for c in ctrls if c.get("label") in {"Yes", "No"}]
        return max(confs) if confs else default

    if field == "claim_type":
        opts = {"Visit", "Procedure", "Medication", "Other"}
        confs = [c.get("conf", 0.0) for c in ctrls if c.get("label") in opts and c.get("selected")]
        return max(confs) if confs else default

    return default


def flags_for_pred(pred: dict[str, Any]) -> list[str]:
    """Generate warning flags for ambiguous predictions."""
    flags = []
    for k in ["Smoker", "Diabetic", "Is this work-related?", "Auto accident?"]:
        if k in pred and pred[k] is None:
            flags.append(f"ambiguous:{k}")
    return flags


def process_input(
    input_path: str,
    out_dir: str,
    conf_threshold: float = 0.5,
    overlays: bool = False,
    llm: bool = False,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Process one or more PDFs through the extraction pipeline."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(input_path)
    files = list(p.rglob("*.pdf")) if p.is_dir() else [p]

    summary = []
    for f in files:
        try:
            res = process_pdf(str(f), dpi=300)
            page = res["pages"][0]
            pred = structure_page(page)

            sex_conf = field_conf_from_controls(page, "sex")
            smoker_conf = field_conf_from_controls(page, "Smoker")
            diabetic_conf = field_conf_from_controls(page, "Diabetic")
            claim_conf = field_conf_from_controls(page, "claim_type")
            flags = flags_for_pred(pred)

            low_conf = []
            if sex_conf and sex_conf < conf_threshold:
                low_conf.append("sex")
            if smoker_conf and smoker_conf < conf_threshold:
                low_conf.append("Smoker")
            if diabetic_conf and diabetic_conf < conf_threshold:
                low_conf.append("Diabetic")
            if claim_conf and claim_conf < conf_threshold:
                low_conf.append("claim_type")
            if low_conf:
                flags.append("low_conf:" + ",".join(low_conf))

            out = {"file": str(f), "pred": pred, "conf": {"sex": sex_conf, "Smoker": smoker_conf, "Diabetic": diabetic_conf, "claim_type": claim_conf}, "flags": flags}

            if llm:
                out = normalize_output(page, out, model=llm_model)

            Path(out_dir, Path(f).stem + "_out.json").write_text(json.dumps(out, indent=2))

            if overlays:
                from extracto.detection.overlay import draw_overlay

                overlay_dir = Path(out_dir) / "overlays"
                overlay_dir.mkdir(parents=True, exist_ok=True)
                draw_overlay(str(f), str(overlay_dir), dpi=300)

            summary.append({
                "file": str(f),
                "sex": pred.get("sex"),
                "sex_conf": round(sex_conf, 3),
                "Smoker": pred.get("Smoker"),
                "Smoker_conf": round(smoker_conf, 3),
                "Diabetic": pred.get("Diabetic"),
                "Diabetic_conf": round(diabetic_conf, 3),
                "claim_type": pred.get("claim_type"),
                "claim_conf": round(claim_conf, 3),
                "allergies_count": len(pred.get("allergies", []) or []),
                "symptoms_count": len(pred.get("symptoms", []) or []),
                "flags": ";".join(flags),
            })
        except Exception as e:
            summary.append({"file": str(f), "error": str(e)})

    # Write summary CSV
    import csv

    csv_path = Path(out_dir, "summary.csv")
    if summary:
        keys = sorted({k for row in summary for k in row.keys()})
        with open(csv_path, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=keys)
            w.writeheader()
            for row in summary:
                w.writerow(row)

    return {"count": len(files), "out_dir": out_dir, "csv": str(csv_path)}


def run_full(
    input_pdf: str,
    out_root: str,
    conf_threshold: float = 0.6,
    overlays: bool = True,
    html: bool = True,
    llm: bool = False,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Full pipeline: split a multi-form PDF, then extract all parts."""
    base = Path(input_pdf).stem
    split_dir = Path(out_root) / "splits" / base
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_pdf(input_pdf, str(split_dir))

    out_dir = Path(out_root) / f"processed_{base}"
    res = process_input(str(split_dir), str(out_dir), conf_threshold=conf_threshold, overlays=overlays, llm=llm, llm_model=llm_model)

    if html:
        try:
            from extracto.pipeline.review import make_review_html

            index = make_review_html(str(out_dir))
            res["html"] = index
        except Exception:
            pass

    summary = {"input": input_pdf, "split_manifest": manifest, "processing": res}
    Path(out_root, f"run_{base}.json").write_text(json.dumps(summary, indent=2))
    return summary
