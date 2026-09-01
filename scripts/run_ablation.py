"""Ablates each evidence signal's contribution, reusing the raw signals already collected by
run_calibration.py / run_pipeline.py — no need to re-run the VLM. For each weight configuration:
recalibrate the threshold on the train split, then measure precision/recall on the untouched
test split. This isolates what each independent evidence source is actually buying the gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidencegate.calibration.nonconformity import EvidenceSignals, nonconformity_score  # noqa: E402
from evidencegate.calibration.risk_control import calibrate_threshold  # noqa: E402
from evidencegate.data import load_family_mask, list_images  # noqa: E402
from evidencegate.evaluation import recall_against_instances  # noqa: E402
from evidencegate.geometry import Box  # noqa: E402

RESULTS_DIR = REPO_ROOT / "assets" / "results"

ABLATIONS = {
    "consensus_only": {"agreement_rate": 0.6, "consensus_tightness": 0.4, "segmentation_agreement": 0.0, "radiometric_plausibility": 0.0},
    "consensus_plus_segmentation": {"agreement_rate": 0.3, "consensus_tightness": 0.2, "segmentation_agreement": 0.5, "radiometric_plausibility": 0.0},
    "consensus_plus_radiometric": {"agreement_rate": 0.3, "consensus_tightness": 0.2, "segmentation_agreement": 0.0, "radiometric_plausibility": 0.5},
    "full_gate": {"agreement_rate": 0.10, "consensus_tightness": 0.10, "segmentation_agreement": 0.40, "radiometric_plausibility": 0.40},
}


def rescored(records: list[dict], weights: dict[str, float]) -> np.ndarray:
    return np.array([nonconformity_score(EvidenceSignals(**r["signals"]), weights) for r in records])


def main() -> None:
    calib_cfg = yaml.safe_load((REPO_ROOT / "configs" / "calibration.yaml").read_text())
    calib_records = json.loads((RESULTS_DIR / "calibration_claims.json").read_text())
    test_records = json.loads((RESULTS_DIR / "test_claims.json").read_text())
    test_images = list_images("test")

    report: dict[str, dict] = {}
    for name, weights in ABLATIONS.items():
        report[name] = {}
        for family in ["red", "bright"]:
            calib_family = [r for r in calib_records if r["family"] == family]
            test_family = [r for r in test_records if r["family"] == family]
            if not calib_family or not test_family:
                continue

            calib_scores = rescored(calib_family, weights)
            calib_labels = np.array([r["is_correct"] for r in calib_family])
            result = calibrate_threshold(calib_scores, calib_labels, calib_cfg["target_risk"], calib_cfg["confidence_delta"])

            test_scores = rescored(test_family, weights)
            accepted_mask = test_scores <= result.threshold
            accepted = [r for r, keep in zip(test_family, accepted_mask) if keep]
            n_accepted = len(accepted)
            n_correct = sum(1 for r in accepted if r["is_correct"])
            precision = n_correct / n_accepted if n_accepted else None

            hit_total, instance_total = 0, 0
            for image in test_images:
                mask = load_family_mask(image.image_id, "test", family, image.load_bgr().shape[:2])
                boxes = [Box(*r["box"]) for r in accepted if r["image_id"] == image.image_id]
                hit, total = recall_against_instances(boxes, mask)
                hit_total += hit
                instance_total += total
            recall = hit_total / instance_total if instance_total else None

            report[name][family] = {
                "calibrated_threshold": result.threshold,
                "certified_risk_bound": result.certified_risk_bound,
                "n_accepted_on_test": n_accepted,
                "measured_precision_on_test": precision,
                "measured_recall_on_test": recall,
            }
            print(f"[{name}/{family}] n_accepted={n_accepted} precision={precision} recall={recall}")

    (RESULTS_DIR / "ablation_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
