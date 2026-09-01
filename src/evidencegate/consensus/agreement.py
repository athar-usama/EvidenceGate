"""Runs K stochastic grounding decodes for one candidate and reduces them to a consensus claim."""

from __future__ import annotations

import random

from PIL import Image

from evidencegate.consensus.clustering import ConsensusCluster, RunHit, dominant_cluster
from evidencegate.geometry import Box, iou
from evidencegate.grounding.florence2 import Grounder
from evidencegate.grounding.perturbations import sample_perturbations
from evidencegate.proposers.red_lesion import Candidate


def run_consensus(
    image: Image.Image,
    candidate: Candidate,
    grounder: Grounder,
    k: int,
    rng: random.Random,
    iou_cluster_threshold: float = 0.3,
) -> ConsensusCluster:
    perturbations = sample_perturbations(image, candidate.box, candidate.lesion_family, k, rng)

    run_hits: list[RunHit] = []
    for run_index, perturbation in enumerate(perturbations):
        crop_local_candidate = Box(
            candidate.box.x1 - perturbation.origin[0],
            candidate.box.y1 - perturbation.origin[1],
            candidate.box.x2 - perturbation.origin[0],
            candidate.box.y2 - perturbation.origin[1],
        )
        hits = grounder.ground(perturbation.crop_image, perturbation.prompt, perturbation.temperature)
        if not hits:
            continue

        best_hit = max(hits, key=lambda h: iou(h.box, crop_local_candidate))
        translated_box = Box(
            best_hit.box.x1 + perturbation.origin[0],
            best_hit.box.y1 + perturbation.origin[1],
            best_hit.box.x2 + perturbation.origin[0],
            best_hit.box.y2 + perturbation.origin[1],
        )
        run_hits.append(RunHit(run_index=run_index, box=translated_box, label=best_hit.label))

    return dominant_cluster(run_hits, iou_threshold=iou_cluster_threshold)
