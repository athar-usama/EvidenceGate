"""Distribution-free threshold calibration via Bonferroni-corrected grid search.

Selects the most permissive nonconformity threshold, among a pre-specified
grid of candidates, whose Clopper-Pearson upper confidence bound on the
false-discovery rate still meets the target risk level — the same statistical
family as split conformal prediction (Vovk, Gammerman & Shafer, 2005) and
Learn-Then-Test risk control (Angelopoulos, Bates, Candès, Jordan & Lei,
2021), simplified to a robust single-parameter search.

An earlier version of this module used literal fixed-sequence testing (walk
candidate thresholds strictest-to-most-permissive, stop at the first one that
fails the bound). That is the textbook Learn-Then-Test procedure, but it
turned out to be fragile here: with `n_grid` candidates tested at a coarse,
roughly evenly-populated grid, a single unlucky small-sample fluctuation at
the first eligible (strictest) threshold — for example, a handful of
accepted claims that happen to all be wrong, purely by chance — permanently
halts the search, even when far more permissive thresholds have plenty of
data and would clear the bound easily. Fixed-sequence testing assumes the
*true* risk is monotone in the threshold, but at finite sample sizes the
*empirical* upper bound is not; testing in a fixed order and stopping on the
first failure inherits that non-monotonicity as spurious total failure.

The fix: evaluate every grid candidate independently (no early stopping) and
correct for the resulting multiple comparisons with a Bonferroni adjustment
(each individual test uses confidence `1 - delta / n_grid` instead of
`1 - delta`), then take the most permissive candidate that still passes. This
controls the same overall (1 - delta) family-wise validity — for a
pre-specified, fixed grid, Bonferroni correction is valid regardless of any
dependence between the tests — while no longer letting one bad grid point
veto every more permissive candidate after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(failures: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if failures == n:
        return 1.0
    return float(beta.ppf(1 - delta, failures + 1, n - failures))


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    target_risk: float
    confidence: float
    n_calibration_claims: int
    n_accepted_at_threshold: int
    n_false_positives_at_threshold: int
    certified_risk_bound: float
    n_grid: int


def _min_accepted_for_zero_failures(target_risk: float, delta: float, margin: int = 5) -> int:
    """Smallest n at which a threshold with zero observed failures could possibly clear the bound.

    Below this n, `clopper_pearson_upper(0, n, delta) > target_risk` always — the bound fails purely
    for lack of data, regardless of the true risk.
    """
    if target_risk <= 0 or target_risk >= 1:
        return 10**6  # degenerate target: no finite n can certify it with a one-sided bound
    n = np.log(delta) / np.log(1 - target_risk)
    return int(np.ceil(n)) + margin


def calibrate_threshold(
    scores: np.ndarray,
    is_correct: np.ndarray,
    target_risk: float = 0.10,
    delta: float = 0.10,
    n_grid: int = 20,
    min_accepted: int | None = None,
) -> CalibrationResult:
    """`scores`: nonconformity scores (lower = more trustworthy). `is_correct`: bool ground-truth labels.

    Candidates are `n_grid` evenly-spaced percentiles of the observed score distribution (so each
    candidate has comparable sample mass) rather than every unique score value. `min_accepted`
    additionally drops any candidate with too little data to be informative regardless of target
    risk; defaults to `_min_accepted_for_zero_failures`.
    """
    scores = np.asarray(scores, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)
    n_total = len(scores)

    if min_accepted is None:
        min_accepted = _min_accepted_for_zero_failures(target_risk, delta / max(1, n_grid))

    fallback_threshold = float(scores.min()) - 1e-9 if n_total else 0.0
    best = CalibrationResult(
        threshold=fallback_threshold,
        target_risk=target_risk,
        confidence=1 - delta,
        n_calibration_claims=n_total,
        n_accepted_at_threshold=0,
        n_false_positives_at_threshold=0,
        certified_risk_bound=0.0,
        n_grid=n_grid,
    )
    if n_total == 0:
        return best

    percentiles = np.linspace(0, 100, n_grid + 1)[1:]  # skip the 0th percentile (accepts nothing)
    candidate_thresholds = np.unique(np.percentile(scores, percentiles))
    bonferroni_delta = delta / max(1, len(candidate_thresholds))

    for tau in candidate_thresholds:
        accepted_mask = scores <= tau
        n_accepted = int(accepted_mask.sum())
        if n_accepted < min_accepted:
            continue

        n_false_positives = int((accepted_mask & ~is_correct).sum())
        ucb = clopper_pearson_upper(n_false_positives, n_accepted, bonferroni_delta)

        if ucb <= target_risk and n_accepted >= best.n_accepted_at_threshold:
            best = CalibrationResult(
                threshold=float(tau),
                target_risk=target_risk,
                confidence=1 - delta,
                n_calibration_claims=n_total,
                n_accepted_at_threshold=n_accepted,
                n_false_positives_at_threshold=n_false_positives,
                certified_risk_bound=ucb,
                n_grid=n_grid,
            )

    return best
