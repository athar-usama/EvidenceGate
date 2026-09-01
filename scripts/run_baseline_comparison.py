"""Naive single-shot grounding baseline: one deterministic Florence-2 call per candidate, no
consensus, no independent verification, no gate. Used purely as the comparison point for the
'what got through, what got caught' hallucination gallery and the headline precision contrast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import yaml
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidencegate.data import load_family_mask, list_images  # noqa: E402
from evidencegate.evaluation import is_correct_claim  # noqa: E402
from evidencegate.geometry import Box, iou  # noqa: E402
from evidencegate.grounding.florence2 import Florence2Grounder  # noqa: E402
from evidencegate.grounding.prompts import prompts_for  # noqa: E402
from evidencegate.pipeline import PROPOSERS  # noqa: E402

RESULTS_DIR = REPO_ROOT / "assets" / "results"


def main() -> None:
    pipeline_cfg = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text())
    calib_cfg = yaml.safe_load((REPO_ROOT / "configs" / "calibration.yaml").read_text())
    grounder = Florence2Grounder(model_id=pipeline_cfg["grounder"]["model_id"])

    records: list[dict] = []
    for image in tqdm(list_images("test"), desc="Naive single-shot baseline on IDRiD test split"):
        bgr = image.load_bgr()
        pil = Image.open(image.image_path).convert("RGB")
        height, width = bgr.shape[:2]

        for family, proposer in PROPOSERS.items():
            mask = load_family_mask(image.image_id, "test", family, (height, width))
            phrase = prompts_for(family)[0]
            for candidate in proposer(bgr):
                padded = candidate.box.expand(0.6, width, height)
                x1, y1, x2, y2 = padded.as_int_tuple()
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = pil.crop((x1, y1, x2, y2))
                hits = grounder.ground(crop, phrase, temperature=0.0)
                if not hits:
                    continue
                local_prior = Box(candidate.box.x1 - x1, candidate.box.y1 - y1, candidate.box.x2 - x1, candidate.box.y2 - y1)
                best = max(hits, key=lambda h: iou(h.box, local_prior))
                claim_box = Box(best.box.x1 + x1, best.box.y1 + y1, best.box.x2 + x1, best.box.y2 + y1)
                correct = is_correct_claim(claim_box, mask, calib_cfg["min_overlap_frac"])
                records.append(
                    {
                        "image_id": image.image_id,
                        "family": family,
                        "box": [claim_box.x1, claim_box.y1, claim_box.x2, claim_box.y2],
                        "candidate_box": [candidate.box.x1, candidate.box.y1, candidate.box.x2, candidate.box.y2],
                        "is_correct": correct,
                        "label": best.label,
                    }
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_claims.json").write_text(json.dumps(records, indent=2))

    for family in ["red", "bright"]:
        family_records = [r for r in records if r["family"] == family]
        n = len(family_records)
        n_correct = sum(1 for r in family_records if r["is_correct"])
        precision = n_correct / n if n else float("nan")
        print(f"[baseline/{family}] n_claims={n} precision={precision:.3f}")


if __name__ == "__main__":
    main()
