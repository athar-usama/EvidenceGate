"""Combines the three independent evidence signals into a single nonconformity score.

Lower score = more conforming = more trustworthy. Each sub-signal is already
normalized to [0, 1]; the combination is a simple weighted average so that the
score stays interpretable (and so the ablation in scripts/run_ablation.py can
zero out individual weights cleanly).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSignals:
    agreement_rate: float  # fraction of K grounding runs that produced this claim
    consensus_tightness: float  # mean pairwise IoU among the runs that agreed
    segmentation_agreement: float  # IoU between the independent segmentation and the consensus box
    radiometric_plausibility: float  # colour/contrast plausibility score for the claimed family


DEFAULT_WEIGHTS = {
    "agreement_rate": 0.10,
    "consensus_tightness": 0.10,
    "segmentation_agreement": 0.40,
    "radiometric_plausibility": 0.40,
}
"""Pilot data on the calibration split showed agreement_rate/consensus_tightness barely separate
correct from incorrect claims for this zero-shot backbone (Florence-2 grounds *something* in
nearly every crop regardless of content), while segmentation_agreement and radiometric_plausibility
show real, consistent separation — weights reflect that rather than the naive equal split. See
docs/METHOD.md and the README's "Peeling back each signal" section for the ablation this is based on."""


def conformity_score(signals: EvidenceSignals, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    total_weight = sum(weights.values())
    score = (
        weights.get("agreement_rate", 0) * signals.agreement_rate
        + weights.get("consensus_tightness", 0) * signals.consensus_tightness
        + weights.get("segmentation_agreement", 0) * signals.segmentation_agreement
        + weights.get("radiometric_plausibility", 0) * signals.radiometric_plausibility
    )
    return score / total_weight if total_weight > 0 else 0.0


def nonconformity_score(signals: EvidenceSignals, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    return 1.0 - conformity_score(signals, weights)
