"""LLM-based post-processing for low-confidence field correction.

Uses OpenAI to review OCR text and suggest minimal corrections
to fields where the CV pipeline has low confidence.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _has_openai_config() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _build_prompt(page_text: str, pred: dict[str, Any], conf: dict[str, float], flags: list[str]) -> str:
    return (
        "You are assisting with deterministic form extraction QA for medical/insurance forms.\n"
        "Input: OCR text lines from a single page, the pipeline's initial prediction (PRED), confidences per field (CONF), and FLAGS.\n"
        "Task: Return a minimal JSON object containing only the fields that should CHANGE in PRED based on textual evidence.\n"
        "Be more assertive when confidence is low (CONF[field] < 0.6) or FLAGS mention 'low_conf:field' or 'ambiguous:field'.\n"
        "Fields: sex (Male/Female/Other), Smoker (true/false), Diabetic (true/false), allergies (string[]), symptoms (string[]), claim_type (Visit/Procedure/Medication/Other).\n"
        "Evidence rules and synonyms:\n"
        "- Sex: look for 'Sex', 'Gender', or 'M/F/O' indicators; map M->Male, F->Female.\n"
        "- Smoker: 'Smoker: Yes/No', 'Non-smoker', 'Tobacco use', 'Never smoker' => false, 'Current every day smoker' => true.\n"
        "- Diabetic: 'Diabetic: Yes/No', 'Diabetes: Type', 'No diabetes' => false.\n"
        "- Allergies: list items; 'No Known Allergies', 'NKA', 'NKDA' => empty array; remove duplicates.\n"
        "- Symptoms: include only from this set if seen in text (case-insensitive): ['Fever','Cough','Headache','Fatigue','Shortness of breath','Nausea','Dizziness','Chest pain']; also accept 'SOB' for 'Shortness of breath'.\n"
        "- Claim Type: map synonyms: 'Consult/Consultation/Visit'->'Visit', 'Procedure/Surgery'->'Procedure', 'Rx/Medication/Prescription'->'Medication', others -> 'Other'.\n"
        "Only output fields that CHANGE from PRED. If you cannot confidently improve a field, omit it. Output must be valid JSON, no commentary.\n\n"
        f"OCR_LINES:\n{page_text}\n\n"
        f"PRED:\n{json.dumps(pred, ensure_ascii=False)}\n\n"
        f"CONF:\n{json.dumps(conf, ensure_ascii=False)}\n\n"
        f"FLAGS:\n{', '.join(flags) if flags else 'none'}\n"
    )


ALLOWED_FIELDS = {"sex", "Smoker", "Diabetic", "allergies", "symptoms", "claim_type"}


def normalize_with_openai(
    page_text: str,
    pred: dict[str, Any],
    conf: dict[str, float],
    flags: list[str],
    model: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call OpenAI to suggest corrections for low-confidence fields."""
    if not _has_openai_config():
        return {}
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = _build_prompt(page_text, pred, conf, flags)
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON with corrected fields; be assertive on low-confidence fields."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=timeout,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`\n ")
            if content.startswith("json\n"):
                content = content[5:]
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_output(page: dict[str, Any], out: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Review and correct low-confidence predictions using LLM.

    Merges LLM suggestions into the prediction, only overriding allowed fields.
    Adds 'llm_adjusted' flag for auditability.
    """
    if not _has_openai_config():
        return out

    lines = page.get("lines", [])
    txt = "\n".join(ln.get("text", "") for ln in lines)
    suggestion = normalize_with_openai(txt, out.get("pred", {}), out.get("conf", {}), out.get("flags", []), model=model)
    if not suggestion:
        return out

    pred = out.get("pred", {}).copy()
    pred.update({k: v for k, v in suggestion.items() if k in ALLOWED_FIELDS})
    out["pred"] = pred
    flags = out.get("flags", [])
    flags.append("llm_adjusted")
    out["flags"] = flags
    return out
