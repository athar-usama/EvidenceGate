"""Calibrates the per-family Verified-tier nonconformity threshold on IDRiD's training split.

Uses IDRiD's own official train/test split rather than an arbitrary re-split,
so the held-out test split used later in run_pipeline.py / make_figures.py has
never been touched during calibration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidencegate.data import load_family_mask, list_images  # noqa: E402
from evidencegate.calibration.nonconformity import DEFAULT_WEIGHTS  # noqa: E402
from evidencegate.calibration.risk_control import calibrate_threshold  # noqa: E402
from evidencegate.evaluation import is_correct_claim  # noqa: E402
from evidencegate.grounding.florence2 import Florence2Grounder  # noqa: E402
from evidencegate.pipeline import process_image  # noqa: E402

RESULTS_DIR = REPO_ROOT / "assets" / "results"


def main() -> None:
    pipeline_cfg = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text())
    calib_cfg = yaml.safe_load((REPO_ROOT / "configs" / "calibration.yaml").read_text())

    grounder = Florence2Grounder(model_id=pipeline_cfg["grounder"]["model_id"])
    k = pipeline_cfg["consensus"]["k_runs"]
    weights = pipeline_cfg["nonconformity_weights"]
    seed = pipeline_cfg.get("seed", 0)

    records: list[dict] = []
    for image in tqdm(list_images("train"), desc="Calibrating on IDRiD train split"):
        bgr_shape = image.load_bgr().shape[:2]
        evidence = process_image(image.image_path, image.image_id, grounder, k=k, seed=seed, weights=weights)
        for claim in evidence.claims:
            mask = load_family_mask(image.image_id, "train", claim.family, bgr_shape)
            correct = is_correct_claim(claim.report_box, mask, calib_cfg["min_overlap_frac"])
            records.append(
                {
                    "image_id": image.image_id,
                    "family": claim.family,
                    "nonconformity": claim.nonconformity,
                    "is_correct": correct,
                    "box": [claim.report_box.x1, claim.report_box.y1, claim.report_box.x2, claim.report_box.y2],
                    "signals": vars(claim.signals),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "calibration_claims.json").write_text(json.dumps(records, indent=2))

    thresholds = {}
    for family in ["red", "bright"]:
        family_records = [r for r in records if r["family"] == family]
        if not family_records:
            continue
        import numpy as np

        scores = np.array([r["nonconformity"] for r in family_records])
        labels = np.array([r["is_correct"] for r in family_records])
        result = calibrate_threshold(scores, labels, calib_cfg["target_risk"], calib_cfg["confidence_delta"])
        thresholds[family] = {
            "threshold": result.threshold,
            "target_risk": result.target_risk,
            "confidence": result.confidence,
            "n_calibration_claims": result.n_calibration_claims,
            "n_accepted_at_threshold": result.n_accepted_at_threshold,
            "n_false_positives_at_threshold": result.n_false_positives_at_threshold,
            "certified_risk_bound": result.certified_risk_bound,
        }
        print(f"[{family}] threshold={result.threshold:.3f} certified_risk<={result.certified_risk_bound:.3f} "
              f"(n_accepted={result.n_accepted_at_threshold}/{result.n_calibration_claims})")

    (RESULTS_DIR / "calibration_thresholds.json").write_text(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
