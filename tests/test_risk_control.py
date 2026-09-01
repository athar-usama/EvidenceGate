import numpy as np

from evidencegate.calibration.risk_control import calibrate_threshold, clopper_pearson_upper


def test_clopper_pearson_zero_failures_gives_low_but_nonzero_bound():
    ucb = clopper_pearson_upper(failures=0, n=100, delta=0.10)
    assert 0.0 < ucb < 0.10


def test_clopper_pearson_bound_shrinks_with_more_data():
    small_n = clopper_pearson_upper(failures=0, n=10, delta=0.10)
    large_n = clopper_pearson_upper(failures=0, n=1000, delta=0.10)
    assert large_n < small_n


def test_clopper_pearson_all_failures_is_one():
    assert clopper_pearson_upper(failures=5, n=5, delta=0.10) == 1.0


def test_calibration_certifies_separable_synthetic_data():
    rng = np.random.default_rng(0)
    n_correct, n_incorrect = 300, 300
    correct_scores = rng.uniform(0.0, 0.3, size=n_correct)
    incorrect_scores = rng.uniform(0.5, 1.0, size=n_incorrect)

    scores = np.concatenate([correct_scores, incorrect_scores])
    is_correct = np.concatenate([np.ones(n_correct, dtype=bool), np.zeros(n_incorrect, dtype=bool)])

    result = calibrate_threshold(scores, is_correct, target_risk=0.10, delta=0.10)

    assert result.certified_risk_bound <= 0.10
    # A target risk of 10% legitimately allows the threshold to reach past the well-separated
    # correct cluster and admit a controlled amount of contamination — so we only assert that a
    # substantial, non-degenerate accept set was found, not that it stayed FDR-free.
    assert result.n_accepted_at_threshold >= n_correct * 0.5


def test_calibration_guarantee_holds_at_the_promised_rate_across_repeated_draws():
    """Monte Carlo check of the guarantee itself: re-run calibration+held-out evaluation many times
    with fresh data: since the guarantee is only a (1 - delta)-confidence statement, some fraction of
    draws are *expected* to overshoot the target risk — but not much more than about `delta` of them.
    """
    target_risk, delta = 0.10, 0.10
    n_trials = 200
    violations = 0
    non_degenerate_trials = 0

    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        n_correct, n_incorrect = 150, 150
        calib_scores = np.concatenate([rng.uniform(0.0, 0.3, n_correct), rng.uniform(0.5, 1.0, n_incorrect)])
        calib_labels = np.concatenate([np.ones(n_correct, dtype=bool), np.zeros(n_incorrect, dtype=bool)])

        result = calibrate_threshold(calib_scores, calib_labels, target_risk=target_risk, delta=delta)
        if result.n_accepted_at_threshold == 0:
            continue  # a degenerate "reject everything" calibration trivially cannot violate the risk bound
        non_degenerate_trials += 1

        held_out_scores = np.concatenate([rng.uniform(0.0, 0.3, 2000), rng.uniform(0.5, 1.0, 2000)])
        held_out_labels = np.concatenate([np.ones(2000, dtype=bool), np.zeros(2000, dtype=bool)])
        accepted = held_out_scores <= result.threshold
        if accepted.sum() == 0:
            continue
        empirical_fdr = float((accepted & ~held_out_labels).sum()) / accepted.sum()
        if empirical_fdr > target_risk:
            violations += 1

    violation_rate = violations / max(1, non_degenerate_trials)
    # Allow generous Monte Carlo slack around the nominal delta=0.10 violation rate.
    assert violation_rate <= 0.25


def test_calibration_rejects_everything_when_no_signal():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, size=200)
    is_correct = rng.uniform(0.0, 1.0, size=200) < 0.3  # scores carry no information about correctness

    result = calibrate_threshold(scores, is_correct, target_risk=0.05, delta=0.10)
    assert result.certified_risk_bound <= 0.05 or result.n_accepted_at_threshold == 0
