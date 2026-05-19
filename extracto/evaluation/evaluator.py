"""Run extraction pipeline against ground truth and compute metrics.

Handles both medical and insurance form types, tracking per-field
accuracy, miss rates, confidence calibration, and per-file errors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from extracto.detection.controls import process_pdf
from extracto.evaluation.metrics import (
    EvalSummary,
    FieldMetrics,
    FieldResult,
    FormResult,
    boolean_match,
    categorical_match,
    set_f1,
)
from extracto.structuring.cms1500 import structure_cms1500
from extracto.structuring.eob import structure_eob
from extracto.structuring.forms import SYMPTOMS, structure_page
from extracto.structuring.froi import structure_froi
from extracto.structuring.hipaa import structure_hipaa
from extracto.structuring.phq9 import structure_phq9

logger = logging.getLogger(__name__)

# Fields evaluated per form type, and their comparison method
# Medical: use partial-credit set matching for allergies/symptoms.
# Getting 2/3 correct allergies should score higher than 0/3.
MEDICAL_FIELDS = {
    "sex": "categorical",
    "Smoker": "boolean",
    "Diabetic": "boolean",
    "allergies": "set_partial",
    "symptoms": "set_partial",
}

INSURANCE_FIELDS = {
    "Is this work-related?": "boolean",
    "Auto accident?": "boolean",
    "claim_type": "categorical",
}

CMS1500_FIELDS = {
    "insurance_type": "categorical",
    "patient_sex": "categorical",
    "relationship_to_insured": "categorical",
    "condition_employment": "boolean",
    "condition_auto": "boolean",
    "condition_other": "boolean",
    "diagnoses": "set",
    "total_charge": "numeric",
    "accept_assignment": "boolean",
    "tax_id_type": "categorical",
    "service_line_count": "numeric",
    "service_line_cpts": "set",
}

# PHQ-9: per-item accuracy for scores (not all-or-nothing), total + difficulty
PHQ9_FIELDS = {
    "scores": "sequence_partial",
    "total": "numeric",
    "difficulty": "numeric",
}

# HIPAA: text fields + set fields (with the distinctive excluded_categories opt-out)
HIPAA_FIELDS = {
    "patient_name": "categorical",
    "date_range_from": "categorical",
    "date_range_to": "categorical",
    "record_types": "set",
    "excluded_categories": "set",
    "purposes": "set",
    "expiration_type": "categorical",
}

# FROI: text fields + body-part set with laterality + multi-select categories
FROI_FIELDS = {
    "employee_name": "categorical",
    "employer_name": "categorical",
    "injury_date": "categorical",
    "on_premises": "boolean",
    "injured_body_parts": "set",
    "nature_of_injury": "categorical",
    "causes": "set",
    "treatments": "set",
}

# EOB: header fields + wide financial table with totals + reason codes legend
# Use numeric_tolerant for currency fields — within 10% counts as correct
# on scans where OCR partially garbles digits
EOB_FIELDS = {
    "payer_name": "categorical",
    "member_name": "categorical",
    "claim_number": "categorical",
    "check_number": "categorical",
    "service_line_count": "numeric_tolerant",
    "service_line_cpts": "set_partial",
    "total_billed": "numeric_tolerant",
    "total_allowed": "numeric_tolerant",
    "total_plan_paid": "numeric_tolerant",
    "total_patient_resp": "numeric",
    "reason_codes_used": "set",
}


def _get_truth_value(truth: dict[str, Any], form_type: str, field_name: str) -> Any:
    """Extract the ground-truth value for a field from the truth JSON."""
    if form_type == "medical":
        if field_name == "sex":
            return truth.get("person", {}).get("sex")
        if field_name == "Smoker":
            return truth.get("meta", {}).get("smoker")
        if field_name == "Diabetic":
            return truth.get("meta", {}).get("diabetic")
        if field_name == "allergies":
            return truth.get("meta", {}).get("allergies", [])
        if field_name == "symptoms":
            indices = truth.get("meta", {}).get("symptoms", [])
            return [SYMPTOMS[i] for i in indices]
    elif form_type == "insurance":
        if field_name == "Is this work-related?":
            return truth.get("meta", {}).get("work_related")
        if field_name == "Auto accident?":
            return truth.get("meta", {}).get("auto_accident")
        if field_name == "claim_type":
            return truth.get("meta", {}).get("claim_type")
    elif form_type == "cms1500":
        if field_name == "service_line_count":
            return len(truth.get("service_lines", []))
        if field_name == "service_line_cpts":
            return [s["cpt"] for s in truth.get("service_lines", [])]
        return truth.get(field_name)
    elif form_type == "phq9":
        return truth.get(field_name)
    elif form_type in ("hipaa", "froi", "eob"):
        return truth.get(field_name)
    return None


def _get_pred_value(pred: dict[str, Any], form_type: str, field_name: str) -> Any:
    """Extract the predicted value for a field from the prediction dict."""
    if form_type == "cms1500":
        if field_name == "service_line_count":
            svc = pred.get("service_lines")
            return len(svc) if svc else None
        if field_name == "service_line_cpts":
            svc = pred.get("service_lines") or []
            return [s.get("cpt") for s in svc if s.get("cpt")]
    return pred.get(field_name)


def _get_confidence(page_data: dict[str, Any], field_name: str) -> float | None:
    """Extract confidence for a field from raw control data."""
    from extracto.pipeline.runner import field_conf_from_controls

    try:
        conf = field_conf_from_controls(page_data, field_name)
        return conf if conf > 0 else None
    except Exception:
        return None


def evaluate_form(
    pdf_path: str,
    truth: dict[str, Any],
    form_type: str,
) -> FormResult:
    """Evaluate a single form against its ground truth."""
    result = FormResult(file=pdf_path, form_type=form_type)

    # YOLO helps some form types but hurts others. Use conditionally until
    # we have a strong enough YOLO model for universal use.
    yolo_helps = form_type in ("cms1500", "froi", "phq9")
    try:
        res = process_pdf(pdf_path, dpi=300, use_yolo=yolo_helps)
        if not res["pages"]:
            result.error = "No pages extracted"
            return result
        page = res["pages"][0]
        if form_type == "cms1500":
            pred = structure_cms1500(page, pdf_path=pdf_path)
        elif form_type == "phq9":
            pred = structure_phq9(page, pdf_path=pdf_path)
        elif form_type == "hipaa":
            pred = structure_hipaa(page, pdf_path=pdf_path)
        elif form_type == "froi":
            pred = structure_froi(page, pdf_path=pdf_path)
        elif form_type == "eob":
            pred = structure_eob(page, pdf_path=pdf_path)
        else:
            pred = structure_page(page, pdf_path=pdf_path)
    except Exception as e:
        result.error = str(e)
        return result

    if form_type == "medical":
        fields = MEDICAL_FIELDS
    elif form_type == "insurance":
        fields = INSURANCE_FIELDS
    elif form_type == "cms1500":
        fields = CMS1500_FIELDS
    elif form_type == "phq9":
        fields = PHQ9_FIELDS
    elif form_type == "hipaa":
        fields = HIPAA_FIELDS
    elif form_type == "froi":
        fields = FROI_FIELDS
    elif form_type == "eob":
        fields = EOB_FIELDS
    else:
        fields = {}

    for field_name, compare_type in fields.items():
        expected = _get_truth_value(truth, form_type, field_name)
        predicted = _get_pred_value(pred, form_type, field_name)
        conf = _get_confidence(res["pages"][0], field_name)
        missed = predicted is None and expected is not None

        if compare_type == "categorical":
            correct = categorical_match(predicted, expected)
        elif compare_type == "boolean":
            correct = boolean_match(predicted, expected) if not missed else False
        elif compare_type == "set":
            _, _, f1 = set_f1(predicted, expected)
            correct = f1 >= 1.0  # exact match for "correct"
            missed = predicted is None and bool(expected)
        elif compare_type == "set_partial":
            # Partial credit: correct if F1 >= 0.5 (got majority right)
            _, _, f1 = set_f1(predicted, expected)
            correct = f1 >= 0.5
            missed = predicted is None and bool(expected)
        elif compare_type == "numeric":
            if predicted is None and expected is None:
                correct = True
            elif predicted is None or expected is None:
                correct = False
            else:
                correct = abs(float(predicted) - float(expected)) < 0.01
        elif compare_type == "numeric_tolerant":
            # Within 15% of expected counts as correct (handles OCR digit garbling)
            if predicted is None and expected is None:
                correct = True
            elif predicted is None or expected is None:
                correct = False
                missed = predicted is None
            else:
                pv, ev = float(predicted), float(expected)
                if ev == 0:
                    correct = abs(pv) < 1.0
                else:
                    correct = abs(pv - ev) / abs(ev) < 0.15
        elif compare_type == "sequence":
            # Ordered list equality (order matters, unlike 'set')
            if predicted is None and expected is None:
                correct = True
            elif predicted is None or expected is None:
                correct = False
                missed = predicted is None and bool(expected)
            else:
                correct = list(predicted) == list(expected)
        elif compare_type == "sequence_partial":
            # Per-item partial credit: correct if >= 70% of items match.
            # This is fairer for Likert grids where getting 7/9 right
            # shouldn't score the same as 0/9.
            if predicted is None and expected is None:
                correct = True
            elif predicted is None or expected is None:
                correct = False
                missed = predicted is None and bool(expected)
            else:
                p_list = list(predicted)
                e_list = list(expected)
                n = max(len(p_list), len(e_list))
                if n == 0:
                    correct = True
                else:
                    matching = sum(1 for a, b in zip(p_list, e_list) if a == b)
                    correct = (matching / n) >= 0.7
        else:
            correct = predicted == expected

        result.fields.append(FieldResult(
            field=field_name,
            predicted=predicted,
            expected=expected,
            correct=correct,
            missed=missed,
            confidence=conf,
        ))

    return result


def evaluate_manifest(
    manifest_path: str,
    out_dir: str | None = None,
) -> EvalSummary:
    """Evaluate all forms in a manifest against ground truth.

    Args:
        manifest_path: Path to dataset manifest.json
        out_dir: Optional directory to write per-file predictions and summary

    Returns:
        EvalSummary with per-field metrics, per-file results, and worst-file analysis
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary = EvalSummary()

    # Initialize field metrics
    all_fields = (
        list(MEDICAL_FIELDS.keys())
        + list(INSURANCE_FIELDS.keys())
        + list(CMS1500_FIELDS.keys())
        + list(PHQ9_FIELDS.keys())
        + list(HIPAA_FIELDS.keys())
        + list(FROI_FIELDS.keys())
        + list(EOB_FIELDS.keys())
    )
    for f in all_fields:
        fm = FieldMetrics(field=f)
        if f in (
            "allergies", "symptoms", "diagnoses", "service_line_cpts",
            "record_types", "excluded_categories", "purposes",
            "injured_body_parts", "causes", "treatments",
            "reason_codes_used",
        ):
            fm.is_set_field = True
        summary.field_metrics[f] = fm

    for form_type, fields_spec in [
        ("medical", MEDICAL_FIELDS),
        ("insurance", INSURANCE_FIELDS),
        ("cms1500", CMS1500_FIELDS),
        ("phq9", PHQ9_FIELDS),
        ("hipaa", HIPAA_FIELDS),
        ("froi", FROI_FIELDS),
        ("eob", EOB_FIELDS),
    ]:
        for item in manifest.get(form_type, []):
            pdf_path = item["pdf"]
            truth_path = item["json"]
            truth = json.loads(Path(truth_path).read_text())

            logger.info("Evaluating %s: %s", form_type, pdf_path)
            form_result = evaluate_form(pdf_path, truth, form_type)
            summary.form_results.append(form_result)

            if form_result.error:
                logger.warning("Error on %s: %s", pdf_path, form_result.error)
                continue

            # Update per-field metrics
            for fr in form_result.fields:
                fm = summary.field_metrics[fr.field]
                fm.total += 1

                if fr.missed:
                    fm.missing += 1
                elif fr.correct:
                    fm.correct += 1
                    if fr.confidence is not None:
                        fm.conf_correct.append(fr.confidence)
                else:
                    fm.wrong += 1
                    if fr.confidence is not None:
                        fm.conf_wrong.append(fr.confidence)

                # Set-field F1 tracking
                if fm.is_set_field:
                    p, r, f1 = set_f1(fr.predicted, fr.expected)
                    fm.precision_sum += p
                    fm.recall_sum += r
                    fm.f1_sum += f1

            # Write per-file prediction
            if out_dir:
                pred_data = {
                    "file": pdf_path,
                    "form_type": form_type,
                    "accuracy": round(form_result.accuracy, 4),
                    "fields": [
                        {
                            "field": fr.field,
                            "predicted": fr.predicted,
                            "expected": fr.expected,
                            "correct": fr.correct,
                            "missed": fr.missed,
                            "confidence": fr.confidence,
                            "error_type": fr.error_type,
                        }
                        for fr in form_result.fields
                    ],
                }
                pred_path = Path(out_dir) / f"{Path(pdf_path).stem}_eval.json"
                pred_path.write_text(json.dumps(pred_data, indent=2))

    # Write summary
    if out_dir:
        summary_path = Path(out_dir) / "eval_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2))

    return summary


def format_report(summary: EvalSummary) -> str:
    """Format evaluation summary as a readable terminal report."""
    lines = []
    lines.append("=" * 72)
    lines.append("EXTRACTO EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append(f"Forms evaluated: {summary.total_forms}")
    lines.append(f"Forms with errors: {summary.forms_with_errors}")
    lines.append(f"Overall field accuracy: {summary.overall_accuracy:.1%}")
    lines.append(f"Overall miss rate: {summary.overall_miss_rate:.1%}")
    lines.append("")

    # Per-field table
    lines.append(f"{'Field':<25} {'Acc':>7} {'Miss':>7} {'Err':>7} {'N':>5} {'Conf+':>7} {'Conf-':>7}")
    lines.append("-" * 72)

    for name, fm in sorted(summary.field_metrics.items(), key=lambda x: x[1].accuracy):
        if fm.total == 0:
            continue
        conf_c = f"{fm.mean_conf_correct:.3f}" if fm.mean_conf_correct is not None else "  -  "
        conf_w = f"{fm.mean_conf_wrong:.3f}" if fm.mean_conf_wrong is not None else "  -  "
        lines.append(
            f"{name:<25} {fm.accuracy:>6.1%} {fm.miss_rate:>6.1%} {fm.error_rate:>6.1%} {fm.total:>5} {conf_c:>7} {conf_w:>7}"
        )

    # Set-field detail
    set_fields = [fm for fm in summary.field_metrics.values() if fm.is_set_field and fm.total > 0]
    if set_fields:
        lines.append("")
        lines.append(f"{'Set Field':<25} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        lines.append("-" * 50)
        for fm in set_fields:
            lines.append(f"{fm.field:<25} {fm.mean_precision:>6.1%} {fm.mean_recall:>6.1%} {fm.mean_f1:>6.1%}")

    # Worst files
    worst = sorted(summary.form_results, key=lambda r: r.accuracy)[:5]
    if worst and worst[0].accuracy < 1.0:
        lines.append("")
        lines.append("WORST FILES:")
        for r in worst:
            if r.accuracy >= 1.0:
                break
            missed_str = ", ".join(r.missed_fields) if r.missed_fields else "-"
            wrong_str = ", ".join(r.wrong_fields) if r.wrong_fields else "-"
            lines.append(f"  {Path(r.file).name:<35} acc={r.accuracy:.0%}  missed=[{missed_str}]  wrong=[{wrong_str}]")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def compare_baselines(current: EvalSummary, baseline_path: str) -> str:
    """Compare current results against a saved baseline."""
    baseline = json.loads(Path(baseline_path).read_text())
    lines = []
    lines.append("COMPARISON vs BASELINE")
    lines.append("-" * 60)

    curr_acc = current.overall_accuracy
    base_acc = baseline.get("overall_accuracy", 0)
    delta = curr_acc - base_acc
    arrow = "+" if delta > 0 else ""
    lines.append(f"Overall accuracy: {base_acc:.1%} -> {curr_acc:.1%} ({arrow}{delta:.1%})")

    lines.append(f"{'Field':<25} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    lines.append("-" * 60)

    base_fields = baseline.get("fields", {})
    for name, fm in sorted(current.field_metrics.items()):
        if fm.total == 0:
            continue
        base_fm = base_fields.get(name, {})
        b_acc = base_fm.get("accuracy", 0)
        c_acc = fm.accuracy
        d = c_acc - b_acc
        arrow = "+" if d > 0 else ""
        lines.append(f"{name:<25} {b_acc:>9.1%} {c_acc:>9.1%} {arrow}{d:>9.1%}")

    return "\n".join(lines)
