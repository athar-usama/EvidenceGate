"""Union-find clustering over K independent grounding decodes.

This is the heart of Consensus-Gated Grounding: LocateAnything treats a box's
four coordinates as one atomic unit to decode together. Here the atomic unit is
raised a level — a *claim* is only admitted once independent decodes of the
same underlying candidate agree with each other in both geometry and
frequency (how many of the K runs produced it at all).

Agreement is judged by IoU *or* normalized center distance, not IoU alone: the
K runs here deliberately vary how much surrounding context the grounder sees
(see grounding/perturbations.py's context ladder), and a wider crop can shift
the model's natural box *size* even when it is looking at the same underlying
feature. Requiring tight IoU across very different fields of view would
penalize scale, not disagreement about location — center distance is the more
scale-robust signal of whether independent views are actually pointing at the
same place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidencegate.geometry import Box, center_distance_norm, iou, mean_box


@dataclass
class RunHit:
    run_index: int
    box: Box
    label: str


@dataclass
class ConsensusCluster:
    hits: list[RunHit] = field(default_factory=list)

    @property
    def consensus_box(self) -> Box:
        return mean_box([h.box for h in self.hits])

    def agreement_rate(self, k_total: int) -> float:
        distinct_runs = len({h.run_index for h in self.hits})
        return distinct_runs / k_total if k_total > 0 else 0.0

    def mean_pairwise_iou(self) -> float:
        boxes = [h.box for h in self.hits]
        if len(boxes) < 2:
            return 1.0 if boxes else 0.0
        total, count = 0.0, 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                total += iou(boxes[i], boxes[j])
                count += 1
        return total / count if count else 0.0

    def mean_pairwise_center_agreement(self) -> float:
        """Scale-robust counterpart to `mean_pairwise_iou` — see module docstring."""
        boxes = [h.box for h in self.hits]
        if len(boxes) < 2:
            return 1.0 if boxes else 0.0
        total, count = 0.0, 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                distance = center_distance_norm(boxes[i], boxes[j])
                total += 1.0 / (1.0 + distance)
                count += 1
        return total / count if count else 0.0

    def top_label(self) -> str:
        if not self.hits:
            return ""
        counts: dict[str, int] = {}
        for h in self.hits:
            counts[h.label] = counts.get(h.label, 0) + 1
        return max(counts, key=counts.get)


def cluster_hits(hits: list[RunHit], iou_threshold: float = 0.3, center_distance_threshold: float = 0.6) -> list[ConsensusCluster]:
    """Greedy single-linkage clustering — small K makes this exact and fast.

    Two hits are joined if they agree on either criterion (tight IoU, for hits from similar
    context scales, or close normalized center distance, robust across very different fields
    of view) — see module docstring.
    """
    n = len(hits)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            agrees = iou(hits[i].box, hits[j].box) >= iou_threshold or (
                center_distance_norm(hits[i].box, hits[j].box) <= center_distance_threshold
            )
            if agrees:
                union(i, j)

    groups: dict[int, list[RunHit]] = {}
    for i, hit in enumerate(hits):
        groups.setdefault(find(i), []).append(hit)

    clusters = [ConsensusCluster(hits=group) for group in groups.values()]
    clusters.sort(key=lambda c: len({h.run_index for h in c.hits}), reverse=True)
    return clusters


def dominant_cluster(hits: list[RunHit], iou_threshold: float = 0.3, center_distance_threshold: float = 0.6) -> ConsensusCluster:
    """The consensus claim for one candidate: its largest-agreement cluster, or an empty one."""
    clusters = cluster_hits(hits, iou_threshold=iou_threshold, center_distance_threshold=center_distance_threshold)
    return clusters[0] if clusters else ConsensusCluster(hits=[])
