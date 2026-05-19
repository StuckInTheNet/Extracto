"""Core metrics computation for form extraction evaluation.

Provides per-field and aggregate metrics with detailed error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldResult:
    """Result for a single field on a single form."""

    field: str
    predicted: Any
    expected: Any
    correct: bool
    missed: bool  # field not predicted at all
    confidence: float | None = None

    @property
    def error_type(self) -> str | None:
        if self.correct:
            return None
        if self.missed:
            return "missing"
        return "wrong"


@dataclass
class FormResult:
    """All field results for a single form."""

    file: str
    form_type: str  # "medical" or "insurance"
    fields: list[FieldResult] = field(default_factory=list)
    error: str | None = None

    @property
    def accuracy(self) -> float:
        if not self.fields:
            return 0.0
        return sum(1 for f in self.fields if f.correct) / len(self.fields)

    @property
    def missed_fields(self) -> list[str]:
        return [f.field for f in self.fields if f.missed]

    @property
    def wrong_fields(self) -> list[str]:
        return [f.field for f in self.fields if f.error_type == "wrong"]


def set_f1(pred: list[str] | None, truth: list[str] | None) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for set-valued fields (allergies, symptoms).

    Returns (precision, recall, f1).
    """
    ps = set(pred or [])
    ts = set(truth or [])
    if not ps and not ts:
        return 1.0, 1.0, 1.0
    if not ps:
        return 0.0, 0.0, 0.0
    if not ts:
        return 0.0, 1.0, 0.0  # predicted items when none expected

    tp = len(ps & ts)
    prec = tp / len(ps)
    rec = tp / len(ts)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def categorical_match(pred: Any, truth: Any) -> bool:
    """Compare categorical values, case-insensitive for strings."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    if isinstance(pred, str) and isinstance(truth, str):
        return pred.lower() == truth.lower()
    return pred == truth


def boolean_match(pred: Any, truth: Any) -> bool:
    """Compare boolean values, treating None as a miss."""
    if pred is None:
        return False
    return bool(pred) == bool(truth)


@dataclass
class FieldMetrics:
    """Aggregate metrics for a single field across all forms."""

    field: str
    total: int = 0
    correct: int = 0
    wrong: int = 0
    missing: int = 0
    # For set-valued fields
    precision_sum: float = 0.0
    recall_sum: float = 0.0
    f1_sum: float = 0.0
    is_set_field: bool = False
    # Confidence tracking
    conf_correct: list[float] = field(default_factory=list)
    conf_wrong: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return self.missing / self.total if self.total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.wrong / self.total if self.total > 0 else 0.0

    @property
    def mean_precision(self) -> float:
        return self.precision_sum / self.total if self.total > 0 and self.is_set_field else 0.0

    @property
    def mean_recall(self) -> float:
        return self.recall_sum / self.total if self.total > 0 and self.is_set_field else 0.0

    @property
    def mean_f1(self) -> float:
        return self.f1_sum / self.total if self.total > 0 and self.is_set_field else 0.0

    @property
    def mean_conf_correct(self) -> float | None:
        return sum(self.conf_correct) / len(self.conf_correct) if self.conf_correct else None

    @property
    def mean_conf_wrong(self) -> float | None:
        return sum(self.conf_wrong) / len(self.conf_wrong) if self.conf_wrong else None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "field": self.field,
            "total": self.total,
            "correct": self.correct,
            "wrong": self.wrong,
            "missing": self.missing,
            "accuracy": round(self.accuracy, 4),
            "miss_rate": round(self.miss_rate, 4),
            "error_rate": round(self.error_rate, 4),
        }
        if self.is_set_field:
            d.update({
                "mean_precision": round(self.mean_precision, 4),
                "mean_recall": round(self.mean_recall, 4),
                "mean_f1": round(self.mean_f1, 4),
            })
        mc = self.mean_conf_correct
        mw = self.mean_conf_wrong
        if mc is not None:
            d["mean_conf_correct"] = round(mc, 4)
        if mw is not None:
            d["mean_conf_wrong"] = round(mw, 4)
        return d


@dataclass
class EvalSummary:
    """Full evaluation summary across all forms."""

    form_results: list[FormResult] = field(default_factory=list)
    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)

    @property
    def total_forms(self) -> int:
        return len(self.form_results)

    @property
    def forms_with_errors(self) -> int:
        return sum(1 for f in self.form_results if f.accuracy < 1.0)

    @property
    def overall_accuracy(self) -> float:
        all_fields = [f for r in self.form_results for f in r.fields]
        if not all_fields:
            return 0.0
        return sum(1 for f in all_fields if f.correct) / len(all_fields)

    @property
    def overall_miss_rate(self) -> float:
        all_fields = [f for r in self.form_results for f in r.fields]
        if not all_fields:
            return 0.0
        return sum(1 for f in all_fields if f.missed) / len(all_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_forms": self.total_forms,
            "forms_with_errors": self.forms_with_errors,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_miss_rate": round(self.overall_miss_rate, 4),
            "fields": {k: v.to_dict() for k, v in self.field_metrics.items()},
            "worst_files": [
                {"file": r.file, "accuracy": round(r.accuracy, 4), "missed": r.missed_fields, "wrong": r.wrong_fields}
                for r in sorted(self.form_results, key=lambda r: r.accuracy)[:10]
            ],
        }
