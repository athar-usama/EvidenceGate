# Method

## 1. Problem statement

A vision-language grounding agent, given an image and a query, returns a set of
localization claims `{(box_i, label_i)}`. In a screening or triage setting, an
unverified claim from a general-purpose, zero-shot grounding model is not
actionable — there is no way to know, at inference time, whether a given box
is evidence of a real finding or a hallucination, short of a human re-checking
every single one. **Consensus-Gated Grounding (CGG)** asks: can we build an
inference-time wrapper around *any* such grounding model that reports a claim
only when it can back it with a statistically defensible guarantee, and stays
silent otherwise?

This is deliberately scoped as an **agent-level trust problem**, not a
detection-accuracy problem. We do not fine-tune, distill, or otherwise touch
the underlying grounding model (Florence-2, used zero-shot). Everything here
sits around it.

## 2. From atomic coordinates to atomic claims

LocateAnything (Wang et al., 2026 — arXiv:2605.27365) decodes a box's four
coordinates as one atomic unit via **Parallel Box Decoding**, rather than as
four sequential tokens, on the grounds that a box is one geometric object and
should be treated as one. CGG applies the same *"decode/verify at the level of
the true atomic unit"* principle one level up the abstraction stack: **a
clinical claim is the atomic unit that matters, not a single model call.** A
claim is only well-formed once several independent probes of the same
underlying evidence agree — geometrically, and across independent modalities
of evidence.

## 3. Algorithm

```
Input:  image I, lesion family f in {red, bright}, grounder G (zero-shot VLM),
        consensus size K, calibrated threshold tau_f

1. Candidates <- ClassicalProposer_f(I)                     # recall-oriented, no learned weights
2. for each candidate c in Candidates:
     a. Perturbations <- SamplePerturbations(c, K)           # crop/scale jitter x prompt paraphrase x temperature
     b. Hits <- { G(crop_k, prompt_k, temp_k) : k = 1..K }   # K independent decodes
     c. Cluster <- DominantCluster(Hits, iou_threshold, center_distance_threshold)  # union-find, IoU OR scale-robust center distance
     d. agreement       <- |distinct runs in Cluster| / K
        tightness        <- mean pairwise scale-robust center agreement within Cluster
        seg_agreement    <- Coverage(GrabCut(claim_box), claim_box)  # independent segmentation tool
        radiometric      <- ColourPlausibility(claim_box, f)       # independent colour/contrast check
     e. nonconformity(c) <- 1 - WeightedAverage(agreement, tightness, seg_agreement, radiometric)
     f. tier(c) <- VERIFIED   if nonconformity(c) <= tau_f
                   FLAGGED    elif agreement > 0
                   ABSTAINED  otherwise
Output: {(c, tier(c), evidence trace(c))}
```

## 4. The statistical guarantee

`tau_f` is not a hand-picked hyperparameter. It is selected on a calibration
split via a **Bonferroni-corrected grid search** — the same distribution-free
statistical family as split conformal prediction (Vovk, Gammerman & Shafer,
2005) and Learn-Then-Test risk control (Angelopoulos, Bates, Candès, Jordan &
Lei, 2021), simplified to a single monotone threshold in a way that is robust
to finite-sample noise (see below):

For a target false-discovery rate `alpha` and confidence `1 - delta`, evaluate
`n_grid` (default 20) candidate thresholds spaced across the observed score
distribution, skipping any that would accept fewer than `min_accepted`
calibration claims (a too-small-n Clopper-Pearson bound is uninformatively
wide regardless of the true risk). At each remaining threshold independently,
compute the Clopper-Pearson upper confidence bound on the empirical
false-discovery rate using a Bonferroni-adjusted confidence level
`1 - delta / n_grid`. Report the most permissive threshold whose bound still
satisfies `UCB <= alpha`.

An earlier version of this procedure used literal *fixed-sequence* testing
(walk thresholds strictest-to-most-permissive, stop at the first failure) —
the textbook Learn-Then-Test recipe. It turned out to be fragile in practice:
a single unlucky small-sample fluctuation at the first eligible (strictest)
threshold permanently halted the search, even when far more permissive
thresholds had ample data and would have cleared the bound easily, because the
*empirical* risk is not actually monotone at finite sample sizes even when the
*true* risk is. The Bonferroni-corrected grid search evaluates every candidate
independently — no early stopping — which controls the same overall
`(1 - delta)` validity for a fixed, pre-specified grid, without letting one bad
grid point veto every better one after it. See `src/evidencegate/calibration/risk_control.py`
for the full reasoning and `tests/test_risk_control.py` for a Monte Carlo check
that the resulting guarantee actually holds at the promised rate across
repeated draws.

**Guarantee.** Under the assumption that calibration and test claims are
exchangeable, for a fresh claim: `P(claim is a true finding | VERIFIED) >= 1 - alpha`,
with confidence `1 - delta` in that statement holding, by construction of the
Clopper-Pearson bound. This is a claim about the *procedure*, not about any
one claim in isolation — the standard reading of a conformal-style guarantee.

`run_pipeline.py` measures whether this holds empirically on IDRiD's official,
untouched test split, and reports the result honestly either way (see
`assets/results/coverage_report.json` and the README's "Does the contract
hold?" section). In practice here: an ambitious `alpha = 0.10` (90% precision)
target was **provably uncertifiable** for either lesion family given the
actual evidence quality achieved — the calibration procedure correctly
reported "no threshold clears this bar" rather than a false one. Sweeping
`alpha` (cheap — it only requires re-running the threshold search over
already-computed scores, not the vision-language model) found `alpha = 0.75`
(a 25% precision floor) as the tightest bar the *bright*-lesion evidence could
support; the *red*-lesion evidence could not support any bar at all, at any
`alpha` tested up to 0.8. Both outcomes are reported in the README exactly as
measured.

## 5. Honest novelty claim

To our knowledge, no existing published work:
- applies distribution-free risk control at the level of **consensus-aggregated
  grounding claims** (rather than at the level of a single model's raw
  confidence or logits) for a VLM localization agent, or
- operationalizes the exact traceable/verifiable medical-grounding-agent gap
  that the LocateAnything authors name as unsolved future work.

This is a synthesis of established building blocks — test-time consensus
ensembling, classical CV as an independent verifier, and Learn-Then-Test risk
control — applied to a gap that a concurrent, unrelated piece of work
explicitly named as open. It is not a claim to have improved on LocateAnything
itself, and it is not a medical device: see Limitations.

## 6. Honest scoping decisions

- **Family-level, not category-level.** The classical proposers distinguish
  *red* lesions (microaneurysms + haemorrhages) from *bright* lesions (hard +
  soft exudates) but not the finer clinical subtype. Ground truth for scoring
  is the union mask per family. A production system would need a stronger
  candidate proposer or a fine-tuned grounder to separate subtypes.
- **Correctness criterion.** A claim counts as correct if at least 15% of its
  box area covers true lesion pixels (`min_overlap_frac` in
  `configs/calibration.yaml`) — a deliberately generous criterion appropriate
  for millimeter-scale lesions with pixel-level but imperfect-boundary ground
  truth, not a tight mAP-style IoU threshold.
- **Signal weights were empirically re-tuned, not assumed.** An initial equal-ish
  weighting (agreement 35%, tightness 20%, segmentation 25%, radiometric 20%)
  was tested against pilot calibration data before any full run; agreement
  rate and consensus tightness showed almost no separation between correct
  and incorrect claims for this zero-shot backbone (it grounds *something* in
  nearly every crop, correct or not) and were down-weighted to 10% each, with
  segmentation and radiometric plausibility raised to 40% each. The ablation
  in the README's "Peeling back each signal" section is this same finding,
  shown quantitatively on the held-out test split.
- **Independence is approximate, not exact.** The segmentation cross-check
  (GrabCut) and the classical proposer (morphological top-hat/black-hat) both
  ultimately look at the same pixels; they are methodologically distinct but
  not statistically independent in a formal sense. The empirical coverage
  check in Section 4 is what actually validates the system, not the
  independence assumption in isolation.

## 7. Limitations

- **The achieved evidence quality supports a real but modest guarantee, not a
  strong one.** The calibration sweep in Section 4 found a certifiable floor
  of 25% precision for bright lesions and none at all for red lesions — a
  ceiling set by the zero-shot backbone and classical CV signals used, not by
  the calibration machinery itself (Section 4's `n_grid` fix specifically
  rules out the alternative explanation that this was a threshold-search
  artifact). A calibrated floor can also numerically undershoot an unaudited
  single-shot baseline's point estimate on a given test set — see the
  README's "Does the contract hold?" section, where this is reported directly
  rather than smoothed over. The value on offer is the guarantee, not the
  highest raw number achievable.
- **Small calibration set.** IDRiD's segmentation subset has 54 calibration
  images and 27 test images — enough to demonstrate the procedure and get a
  real, honestly-reported answer, not enough to make a population-level
  clinical claim. Confidence intervals on the measured test precision are
  reported for exactly this reason.
- **Claim-level, not image-level, exchangeability.** Split conformal / LTT
  guarantees assume exchangeable calibration/test units. Claims from the same
  image are not independent of each other (shared image-level nuisance
  factors: illumination, camera, patient). The guarantee is stated and
  evaluated at the claim level, consistent with common practice in the
  conformal-prediction-for-detection literature, but this is a known
  simplification worth flagging rather than hiding.
- **Zero-shot VLM ceiling.** Florence-2 was never trained on fundus imagery.
  The gate can only certify what the underlying grounder is capable of
  proposing in the first place — it raises precision of what gets reported,
  it cannot manufacture recall the base model doesn't have.
- **Not a diagnostic device.** This is a research artifact demonstrating a
  trust-calibration mechanism, evaluated against public segmentation ground
  truth. It has no clinical validation and makes no diagnostic claims.
- **Supervised fine-tuning, not RL.** Per the LocateAnything authors' own
  stated future work, an RL-tuned or purpose-built grounder would likely raise
  the ceiling on what this gate has to work with; that is out of scope here.

## 8. References

- Wang, S., Liu, S., Kuang, Y., Wei, X., Liu, Y., Li, Z., Man, Y., Chen, G.,
  Tao, A., Liu, G., Kautz, J., Zhang, L., & Yu, Z. *LocateAnything: Fast and
  High-Quality Vision-Language Grounding with Parallel Box Decoding.*
  arXiv:2605.27365, 2026.
- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L.
  *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk
  Control.* arXiv:2110.01052, 2021.
- Vovk, V., Gammerman, A., & Shafer, G. *Algorithmic Learning in a Random
  World.* Springer, 2005.
- Porwal, P., Pachade, S., Kamble, R., Kokare, M., Deshmukh, G.,
  Sahasrabuddhe, V., & Meriaudeau, F. *Indian Diabetic Retinopathy Image
  Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research.*
  Data, 3(3), 25, 2018. (CC BY 4.0)
- Xiao, B. et al. *Florence-2: Advancing a Unified Representation for a
  Variety of Vision Tasks.* CVPR, 2024.
