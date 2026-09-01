from evidencegate.consensus.clustering import RunHit, cluster_hits, dominant_cluster
from evidencegate.geometry import Box


def _hit(run_index: int, x1: float, label: str = "lesion") -> RunHit:
    return RunHit(run_index=run_index, box=Box(x1, 0, x1 + 10, 10), label=label)


def test_agreeing_hits_form_one_cluster():
    hits = [_hit(0, 0), _hit(1, 1), _hit(2, 0.5)]
    clusters = cluster_hits(hits, iou_threshold=0.3)
    assert len(clusters) == 1
    assert clusters[0].agreement_rate(k_total=3) == 1.0


def test_disagreeing_hits_form_separate_clusters():
    hits = [_hit(0, 0), _hit(1, 100)]
    clusters = cluster_hits(hits, iou_threshold=0.3)
    assert len(clusters) == 2


def test_dominant_cluster_picks_highest_agreement():
    hits = [_hit(0, 0), _hit(1, 1), _hit(2, 100)]
    dominant = dominant_cluster(hits, iou_threshold=0.3)
    assert dominant.agreement_rate(k_total=3) - (2 / 3) < 1e-9


def test_empty_hits_yields_empty_cluster():
    dominant = dominant_cluster([], iou_threshold=0.3)
    assert dominant.hits == []
    assert dominant.agreement_rate(k_total=8) == 0.0


def test_duplicate_hits_from_same_run_count_once_toward_agreement():
    hits = [_hit(0, 0), _hit(0, 1)]  # same run, two boxes that happen to overlap
    clusters = cluster_hits(hits, iou_threshold=0.3)
    assert clusters[0].agreement_rate(k_total=1) == 1.0
