"""Tests for evaluation metrics computation."""

from extracto.evaluation.metrics import (
    EvalSummary,
    FieldMetrics,
    FieldResult,
    FormResult,
    boolean_match,
    categorical_match,
    set_f1,
)


class TestSetF1:
    def test_perfect_match(self):
        p, r, f1 = set_f1(["a", "b"], ["a", "b"])
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_partial_match(self):
        p, r, f1 = set_f1(["a", "b", "c"], ["a", "b"])
        assert p == 2 / 3
        assert r == 1.0
        assert 0.79 < f1 < 0.81  # 2 * (2/3) * 1 / (2/3 + 1) = 0.8

    def test_no_overlap(self):
        p, r, f1 = set_f1(["a"], ["b"])
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_both_empty(self):
        p, r, f1 = set_f1([], [])
        assert f1 == 1.0

    def test_pred_empty_truth_nonempty(self):
        p, r, f1 = set_f1([], ["a"])
        assert f1 == 0.0

    def test_none_inputs(self):
        p, r, f1 = set_f1(None, None)
        assert f1 == 1.0

    def test_pred_none_truth_present(self):
        p, r, f1 = set_f1(None, ["a", "b"])
        assert f1 == 0.0


class TestCategoricalMatch:
    def test_exact_match(self):
        assert categorical_match("Male", "Male")

    def test_case_insensitive(self):
        assert categorical_match("male", "Male")

    def test_mismatch(self):
        assert not categorical_match("Male", "Female")

    def test_both_none(self):
        assert categorical_match(None, None)

    def test_one_none(self):
        assert not categorical_match(None, "Male")
        assert not categorical_match("Male", None)


class TestBooleanMatch:
    def test_true_true(self):
        assert boolean_match(True, True)

    def test_false_false(self):
        assert boolean_match(False, False)

    def test_mismatch(self):
        assert not boolean_match(True, False)

    def test_none_is_miss(self):
        assert not boolean_match(None, True)
        assert not boolean_match(None, False)


class TestFieldResult:
    def test_correct_has_no_error(self):
        fr = FieldResult(field="sex", predicted="Male", expected="Male", correct=True, missed=False)
        assert fr.error_type is None

    def test_wrong_error_type(self):
        fr = FieldResult(field="sex", predicted="Female", expected="Male", correct=False, missed=False)
        assert fr.error_type == "wrong"

    def test_missing_error_type(self):
        fr = FieldResult(field="sex", predicted=None, expected="Male", correct=False, missed=True)
        assert fr.error_type == "missing"


class TestFormResult:
    def test_accuracy(self):
        fr = FormResult(file="test.pdf", form_type="medical", fields=[
            FieldResult("sex", "Male", "Male", True, False),
            FieldResult("Smoker", True, False, False, False),
            FieldResult("Diabetic", None, True, False, True),
        ])
        assert abs(fr.accuracy - 1 / 3) < 0.01

    def test_missed_and_wrong_fields(self):
        fr = FormResult(file="test.pdf", form_type="medical", fields=[
            FieldResult("sex", "Male", "Male", True, False),
            FieldResult("Smoker", True, False, False, False),
            FieldResult("Diabetic", None, True, False, True),
        ])
        assert fr.missed_fields == ["Diabetic"]
        assert fr.wrong_fields == ["Smoker"]


class TestFieldMetrics:
    def test_accuracy_and_rates(self):
        fm = FieldMetrics(field="sex", total=10, correct=7, wrong=2, missing=1)
        assert fm.accuracy == 0.7
        assert fm.miss_rate == 0.1
        assert fm.error_rate == 0.2

    def test_empty(self):
        fm = FieldMetrics(field="sex")
        assert fm.accuracy == 0.0
        assert fm.miss_rate == 0.0

    def test_confidence_tracking(self):
        fm = FieldMetrics(field="sex", total=4, correct=3, wrong=1)
        fm.conf_correct = [0.9, 0.8, 0.95]
        fm.conf_wrong = [0.3]
        assert abs(fm.mean_conf_correct - 0.883) < 0.01
        assert fm.mean_conf_wrong == 0.3

    def test_set_field_metrics(self):
        fm = FieldMetrics(field="allergies", total=3, correct=2, wrong=1, is_set_field=True)
        fm.precision_sum = 2.5
        fm.recall_sum = 2.0
        fm.f1_sum = 2.2
        assert abs(fm.mean_precision - 0.833) < 0.01
        assert abs(fm.mean_recall - 0.667) < 0.01
        assert abs(fm.mean_f1 - 0.733) < 0.01

    def test_to_dict(self):
        fm = FieldMetrics(field="sex", total=10, correct=8, wrong=1, missing=1)
        d = fm.to_dict()
        assert d["field"] == "sex"
        assert d["accuracy"] == 0.8
        assert d["miss_rate"] == 0.1


class TestEvalSummary:
    def test_overall_accuracy(self):
        s = EvalSummary()
        s.form_results = [
            FormResult("a.pdf", "medical", [
                FieldResult("sex", "M", "M", True, False),
                FieldResult("Smoker", True, True, True, False),
            ]),
            FormResult("b.pdf", "medical", [
                FieldResult("sex", "F", "M", False, False),
                FieldResult("Smoker", None, True, False, True),
            ]),
        ]
        assert s.overall_accuracy == 0.5
        assert s.overall_miss_rate == 0.25
        assert s.forms_with_errors == 1

    def test_to_dict_has_worst_files(self):
        s = EvalSummary()
        s.form_results = [
            FormResult("good.pdf", "medical", [FieldResult("sex", "M", "M", True, False)]),
            FormResult("bad.pdf", "medical", [FieldResult("sex", "F", "M", False, False)]),
        ]
        s.field_metrics = {"sex": FieldMetrics("sex", total=2, correct=1, wrong=1)}
        d = s.to_dict()
        assert d["worst_files"][0]["file"] == "bad.pdf"
