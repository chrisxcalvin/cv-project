from core.active_challenge import _ear_from_points, _mar_from_points


def test_ear_from_points_wide_open_eye_has_higher_ear_than_closed():
    # Horizontal corners fixed at (0,0)-(10,0); vary vertical gap only.
    open_eye = [(0, 0), (3, -4), (7, -4), (10, 0), (7, 4), (3, 4)]
    closed_eye = [(0, 0), (3, -0.2), (7, -0.2), (10, 0), (7, 0.2), (3, 0.2)]

    open_ear = _ear_from_points(open_eye)
    closed_ear = _ear_from_points(closed_eye)

    assert open_ear > closed_ear


def test_ear_from_points_degenerate_horizontal_distance_returns_zero():
    pts = [(5, 0), (5, -1), (5, -1), (5, 0), (5, 1), (5, 1)]
    assert _ear_from_points(pts) == 0.0


def test_mar_from_points_open_mouth_has_higher_mar_than_closed():
    # pts = [left_corner, upper_inner_lip, right_corner, lower_inner_lip]
    open_mouth = [(0, 0), (5, -8), (10, 0), (5, 8)]
    closed_mouth = [(0, 0), (5, -0.5), (10, 0), (5, 0.5)]

    assert _mar_from_points(open_mouth) > _mar_from_points(closed_mouth)


def test_mar_from_points_degenerate_horizontal_distance_returns_zero():
    pts = [(5, 0), (5, -3), (5, 0), (5, 3)]
    assert _mar_from_points(pts) == 0.0
