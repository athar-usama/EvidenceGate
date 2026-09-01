"""Renders every image asset used in the README from the already-computed pipeline results.

Re-runs the gated pipeline on a curated handful of test images (needed to get
full Claim objects with per-run hits and segmentation masks — the summary
JSONs from run_pipeline.py only keep the aggregated numbers) and pairs them
against the naive-baseline JSON to build the hallucination gallery.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidencegate.data import list_images  # noqa: E402
from evidencegate.geometry import Box, iou  # noqa: E402
from evidencegate.grounding.florence2 import Florence2Grounder  # noqa: E402
from evidencegate.pipeline import TIER_ABSTAINED, TIER_FLAGGED, TIER_VERIFIED, process_image  # noqa: E402
from evidencegate.report.evidence_card import render_evidence_card  # noqa: E402
from evidencegate.report.gallery import add_caption, annotate_image, make_grid, side_by_side  # noqa: E402
from evidencegate.report.gauge_svg import write_trust_ring  # noqa: E402

RESULTS_DIR = REPO_ROOT / "assets" / "results"
FIGURES_DIR = REPO_ROOT / "assets" / "figures"
CARDS_DIR = REPO_ROOT / "assets" / "cards"

GALLERY_IMAGE_IDS = [f"IDRiD_{i:02d}" for i in [55, 58, 60, 63, 66, 68, 70, 74, 78]]


def main() -> None:
    pipeline_cfg = yaml.safe_load((REPO_ROOT / "configs" / "pipeline.yaml").read_text())
    thresholds_raw = json.loads((RESULTS_DIR / "calibration_thresholds.json").read_text())
    verified_thresholds = {family: entry["threshold"] for family, entry in thresholds_raw.items()}
    baseline_claims = json.loads((RESULTS_DIR / "baseline_claims.json").read_text())

    grounder = Florence2Grounder(model_id=pipeline_cfg["grounder"]["model_id"])
    k = pipeline_cfg["consensus"]["k_runs"]
    weights = pipeline_cfg["nonconformity_weights"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    by_id = {img.image_id: img for img in list_images("test")}
    gallery_thumbs, trace_thumbs = [], []
    all_claims_by_image = {}

    for image_id in GALLERY_IMAGE_IDS:
        if image_id not in by_id:
            continue
        image = by_id[image_id]
        bgr = image.load_bgr()
        evidence = process_image(
            image.image_path, image_id, grounder, k=k, seed=0,
            verified_thresholds=verified_thresholds, weights=weights,
        )
        all_claims_by_image[image_id] = (bgr, evidence.claims)

        verdict = annotate_image(bgr, evidence.claims, tiers_to_show=(TIER_VERIFIED, TIER_FLAGGED))
        gallery_thumbs.append(add_caption(verdict, image_id))

        trace = annotate_image(bgr, evidence.claims, tiers_to_show=(TIER_VERIFIED, TIER_FLAGGED, TIER_ABSTAINED))
        trace_thumbs.append(add_caption(trace, image_id))

    make_grid(gallery_thumbs, cols=3, cell_size=300).save(FIGURES_DIR / "verdict_gallery.png")
    make_grid(trace_thumbs, cols=3, cell_size=300).save(FIGURES_DIR / "full_trace_gallery.png")

    # Flagship evidence cards: best verified, a flagged, and an abstained claim, one per family.
    for family in ["red", "bright"]:
        pool = [
            (bgr, claim)
            for bgr, claims in all_claims_by_image.values()
            for claim in claims
            if claim.family == family
        ]
        verified_pool = sorted((c for c in pool if c[1].tier == TIER_VERIFIED), key=lambda x: x[1].nonconformity)
        flagged_pool = [c for c in pool if c[1].tier == TIER_FLAGGED]
        abstained_pool = [c for c in pool if c[1].tier == TIER_ABSTAINED and c[1].consensus.hits]

        if verified_pool:
            render_evidence_card(*verified_pool[0], out_path=str(CARDS_DIR / f"{family}_verified_example.png"))
        if flagged_pool:
            render_evidence_card(*flagged_pool[0], out_path=str(CARDS_DIR / f"{family}_flagged_example.png"))
        if abstained_pool:
            render_evidence_card(*abstained_pool[0], out_path=str(CARDS_DIR / f"{family}_abstained_example.png"))

        if verified_pool:
            conformity = 1 - verified_pool[0][1].nonconformity
            write_trust_ring(str(FIGURES_DIR / f"{family}_trust_ring.svg"), conformity, TIER_VERIFIED, label="VERIFIED")

    # Hallucination gallery: baseline claims that were WRONG, paired with what the gate did with the same candidate.
    pairs = []
    for base in baseline_claims:
        if base["is_correct"]:
            continue
        image_id = base["image_id"]
        if image_id not in all_claims_by_image:
            continue
        bgr, claims = all_claims_by_image[image_id]
        base_candidate = Box(*base["candidate_box"])
        match = max(
            (c for c in claims if c.family == base["family"]),
            key=lambda c: iou(c.candidate.box, base_candidate),
            default=None,
        )
        if match is not None and iou(match.candidate.box, base_candidate) > 0.9:
            pairs.append((bgr, base, match))

    rng = random.Random(0)
    rng.shuffle(pairs)
    hallucination_rows = []
    for bgr, base, gated_claim in pairs[:6]:
        region = Box(*base["candidate_box"]).expand(2.0, bgr.shape[1], bgr.shape[0])
        x1, y1, x2, y2 = region.as_int_tuple()
        import cv2

        crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        base_crop = Image.fromarray(crop_rgb).copy()
        gated_crop = Image.fromarray(crop_rgb).copy()

        from PIL import ImageDraw

        bx = Box(*base["box"])
        ImageDraw.Draw(base_crop).rectangle(
            [bx.x1 - x1, bx.y1 - y1, bx.x2 - x1, bx.y2 - y1], outline=(220, 40, 40), width=4
        )
        gx = gated_claim.report_box
        gated_color = {"verified": (26, 152, 80), "flagged": (217, 164, 6), "abstained": (192, 57, 43)}[gated_claim.tier]
        ImageDraw.Draw(gated_crop).rectangle(
            [gx.x1 - x1, gx.y1 - y1, gx.x2 - x1, gx.y2 - y1], outline=gated_color, width=4
        )

        pair_image = side_by_side(
            base_crop, gated_crop, "single-shot VLM (wrong)", f"consensus-gated -> {gated_claim.tier.upper()}"
        )
        hallucination_rows.append(pair_image)

    if hallucination_rows:
        make_grid(hallucination_rows, cols=2, cell_size=360).save(FIGURES_DIR / "hallucination_gallery.png")

    print(f"Wrote figures to {FIGURES_DIR} and cards to {CARDS_DIR}")


if __name__ == "__main__":
    main()
