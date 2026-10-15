from elt_rlvr.grading import GradeReport, ModelGrade, RawTableGrade
from elt_rlvr.reward import RewardConfig, compute_reward


def _report(el_ok, models, submitted=True):
    raw = [RawTableGrade("t1", 10, True, 10 if el_ok else 9, el_ok, True)]
    return GradeReport("x", raw, models, True, submitted)


def test_models_only_count_after_complete_el():
    good = [ModelGrade("m", True, True, matched=["a", "b"])]
    r, _ = compute_reward(_report(False, good), RewardConfig(el_weight=0.2))
    assert r == 0.0
    r, _ = compute_reward(_report(True, good), RewardConfig(el_weight=0.2))
    assert r == 1.0


def test_column_credit_is_capped_below_full_match():
    half = [ModelGrade("m", True, False, matched=["a"], unmatched=["b"])]
    r, _ = compute_reward(_report(True, half), RewardConfig(el_weight=0.2, column_credit=0.5))
    assert abs(r - (0.2 + 0.8 * 0.25)) < 1e-9
    r, _ = compute_reward(_report(True, half), RewardConfig(mode="model_fraction", el_weight=0.2))
    assert abs(r - 0.2) < 1e-9


def test_unsubmitted_is_zero_regardless():
    good = [ModelGrade("m", True, True, matched=["a"])]
    assert compute_reward(_report(True, good, submitted=False))[0] == 0.0
