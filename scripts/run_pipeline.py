"""Runs the fully calibrated agent on IDRiD's held-out test split and measures the real coverage.

This is the split that never touched calibration, so the numbers this script
produces are the honest answer to "does the contract actually hold?".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import beta
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidencegate.data import load_family_mask, list_images  # noqa: E402
from evidencegate.evaluation import is_correct_claim, recall_against_instances  # noqa: E402
from evidencegate.geometry import Box  # noqa: E402
from evidencegate.grounding.florence2 import Florence2Grounder  # noqa: E402
from evidencegate.pipeline import TIER_VERIFIED, process_image  # noqa: E402

RESULTS_DIR = REPO_ROOT / "assets" / "results"


def clopper_pearson_interval(successes: int, n: int, confidence: float = 0.90) -> tuple[float, float]:
    alpha = 1 - confidence
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    upper = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lower), float(upper)


def main() -> None:
    pipeline_cfg = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text())
    calib_cfg = yaml.safe_load((REPO_ROOT / "configs" / "calibration.yaml").read_text())
    thresholds_raw = json.loads((RESULTS_DIR / "calibration_thresholds.json").read_text())
    verified_thresholds = {family: entry["threshold"] for family, entry in thresholds_raw.items()}

    grounder = Florence2Grounder(model_id=pipeline_cfg["grounder"]["model_id"])
    k = pipeline_cfg["consensus"]["k_runs"]
    weights = pipeline_cfg["nonconformity_weights"]
    seed = pipeline_cfg.get("seed", 0)

    all_records: list[dict] = []
    for image in tqdm(list_images("test"), desc="Running gated agent on IDRiD test split"):
        bgr_shape = image.load_bgr().shape[:2]
        evidence = process_image(
            image.image_path, image.image_id, grounder, k=k, seed=seed,
            verified_thresholds=verified_thresholds, weights=weights,
        )
        for claim in evidence.claims:
            mask = load_family_mask(image.image_id, "test", claim.family, bgr_shape)
            correct = is_correct_claim(claim.report_box, mask, calib_cfg["min_overlap_frac"])
            all_records.append(
                {
                    "image_id": image.image_id,
                    "family": claim.family,
                    "tier": claim.tier,
                    "nonconformity": claim.nonconformity,
                    "is_correct": correct,
                    "box": [claim.report_box.x1, claim.report_box.y1, claim.report_box.x2, claim.report_box.y2],
                    "candidate_box": [claim.candidate.box.x1, claim.candidate.box.y1, claim.candidate.box.x2, claim.candidate.box.y2],
                    "signals": vars(claim.signals),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "test_claims.json").write_text(json.dumps(all_records, indent=2))

    test_images = list_images("test")
    coverage_report = {}
    for family in ["red", "bright"]:
        verified = [r for r in all_records if r["family"] == family and r["tier"] == TIER_VERIFIED]
        n = len(verified)
        n_correct = sum(1 for r in verified if r["is_correct"])
        precision = n_correct / n if n else None
        ci_low, ci_high = clopper_pearson_interval(n_correct, n, 0.90) if n else (None, None)

        hit_total, instance_total = 0, 0
        for image in test_images:
            mask = load_family_mask(image.image_id, "test", family, image.load_bgr().shape[:2])
            boxes = [
                Box(*r["box"])
                for r in verified
                if r["image_id"] == image.image_id
            ]
            hit, total = recall_against_instances(boxes, mask)
            hit_total += hit
            instance_total += total
        recall = hit_total / instance_total if instance_total else None

        target_risk = thresholds_raw.get(family, {}).get("target_risk")
        coverage_report[family] = {
            "target_risk": target_risk,
            "certified_confidence": thresholds_raw.get(family, {}).get("confidence"),
            "n_verified_claims_on_test": n,
            "n_correct_of_verified": n_correct,
            "measured_precision": precision,
            "measured_precision_90pct_CI": [ci_low, ci_high],
            "contract_held": (1 - precision) <= target_risk + 1e-6 if precision is not None and target_risk is not None else None,
            "recall_of_true_lesion_instances": recall,
            "n_lesion_instances_hit": hit_total,
            "n_lesion_instances_total": instance_total,
        }

    (RESULTS_DIR / "coverage_report.json").write_text(json.dumps(coverage_report, indent=2))
    print(json.dumps(coverage_report, indent=2))


if __name__ == "__main__":
    main()
