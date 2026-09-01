from evidencegate.geometry import Box, center_distance_norm, iou, mean_box, union_box


def test_iou_identical_boxes_is_one():
    a = Box(0, 0, 10, 10)
    assert iou(a, a) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = Box(0, 0, 10, 10)
    b = Box(20, 20, 30, 30)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = Box(0, 0, 10, 10)
    b = Box(5, 0, 15, 10)
    # intersection 5x10=50, union 100+100-50=150
    assert abs(iou(a, b) - 50 / 150) < 1e-9


def test_expand_grows_symmetrically_and_clips():
    a = Box(10, 10, 20, 20)
    expanded = a.expand(1.0, width=100, height=100)
    assert expanded.x1 == 0.0 and expanded.y1 == 0.0
    assert expanded.x2 == 30.0 and expanded.y2 == 30.0

    near_edge = Box(0, 0, 5, 5)
    clipped = near_edge.expand(2.0, width=100, height=100)
    assert clipped.x1 == 0.0 and clipped.y1 == 0.0


def test_union_and_mean_box():
    boxes = [Box(0, 0, 10, 10), Box(5, 5, 15, 15)]
    u = union_box(boxes)
    assert (u.x1, u.y1, u.x2, u.y2) == (0, 0, 15, 15)
    m = mean_box(boxes)
    assert (m.x1, m.y1, m.x2, m.y2) == (2.5, 2.5, 12.5, 12.5)


def test_center_distance_norm_zero_for_concentric_boxes():
    a = Box(0, 0, 10, 10)
    b = Box(2, 2, 8, 8)
    assert center_distance_norm(a, b) == 0.0
