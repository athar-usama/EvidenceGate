"""The agent itself: proposal -> consensus grounding -> independent verification -> gating.

Mirrors the five-stage medical-image agent sketched in the source LinkedIn post
(interpret query -> locate structures -> extract native-resolution regions ->
invoke segmentation/measurement tools -> answer linked to evidence), with the
addition that step 5 here comes with a calibrated statistical guarantee rather
than a bare model output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
from PIL import Image

from evidencegate.calibration.nonconformity import DEFAULT_WEIGHTS, EvidenceSignals, nonconformity_score
from evidencegate.consensus.agreement import run_consensus
from evidencegate.consensus.clustering import ConsensusCluster
from evidencegate.geometry import Box
from evidencegate.grounding.florence2 import Grounder
from evidencegate.proposers.bright_lesion import propose_bright_lesions
from evidencegate.proposers.red_lesion import Candidate, propose_red_lesions
from evidencegate.verification.radiometry import plausibility_score
from evidencegate.verification.segmentation import SegmentationResult, segment_claim

TIER_VERIFIED = "verified"
TIER_FLAGGED = "flagged"
TIER_ABSTAINED = "abstained"


@dataclass
class Claim:
    family: str
    candidate: Candidate
    consensus: ConsensusCluster
    segmentation: SegmentationResult
    signals: EvidenceSignals
    nonconformity: float
    k_runs: int
    tier: str = TIER_ABSTAINED

    @property
    def report_box(self) -> Box:
        return self.consensus.consensus_box if self.consensus.hits else self.candidate.box


@dataclass
class ImageEvidence:
    image_id: str
    width: int
    height: int
    claims: list[Claim] = field(default_factory=list)


PROPOSERS = {"red": propose_red_lesions, "bright": propose_bright_lesions}


def build_claim(
    bgr,
    candidate: Candidate,
    image_pil: Image.Image,
    grounder: Grounder,
    k: int,
    rng: random.Random,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> Claim:
    consensus = run_consensus(image_pil, candidate, grounder, k, rng)
    claim_region = consensus.consensus_box if consensus.hits else candidate.box
    seg = segment_claim(bgr, claim_region)
    radiometric = plausibility_score(bgr, claim_region, candidate.lesion_family)

    signals = EvidenceSignals(
        agreement_rate=consensus.agreement_rate(k) if consensus.hits else 0.0,
        consensus_tightness=consensus.mean_pairwise_center_agreement() if consensus.hits else 0.0,
        segmentation_agreement=seg.agreement,
        radiometric_plausibility=radiometric,
    )
    score = nonconformity_score(signals, weights)

    return Claim(
        family=candidate.lesion_family,
        candidate=candidate,
        consensus=consensus,
        segmentation=seg,
        signals=signals,
        nonconformity=score,
        k_runs=k,
    )


def tier_claim(claim: Claim, verified_threshold: float) -> str:
    if claim.nonconformity <= verified_threshold:
        return TIER_VERIFIED
    if claim.consensus.hits:
        return TIER_FLAGGED
    return TIER_ABSTAINED


def process_image(
    image_path,
    image_id: str,
    grounder: Grounder,
    k: int = 8,
    seed: int = 0,
    verified_thresholds: dict[str, float] | None = None,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> ImageEvidence:
    bgr = cv2.imread(str(image_path))
    image_pil = Image.open(image_path).convert("RGB")
    height, width = bgr.shape[:2]
    rng = random.Random(seed)

    evidence = ImageEvidence(image_id=image_id, width=width, height=height)
    for family, proposer in PROPOSERS.items():
        candidates = proposer(bgr)
        for candidate in candidates:
            claim = build_claim(bgr, candidate, image_pil, grounder, k, rng, weights)
            if verified_thresholds is not None:
                claim.tier = tier_claim(claim, verified_thresholds.get(family, 0.0))
            evidence.claims.append(claim)

    return evidence
