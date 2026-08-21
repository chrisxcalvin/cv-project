"""
Pins the mirrored-camera yaw sign convention documented in
core/active_challenge.py::_compute_yaw and used by both
evaluate_single_frame() (Streamlit path) and run_challenge() (--live path).

Without this, the required_sign / frame_ok comparisons in those two methods
could silently flip (e.g. during a refactor) and turn_left/turn_right would
swap in the demo with no test catching it.

Landmarks are faked via monkeypatching ActiveChallenge._get_landmarks, so
these tests don't depend on mediapipe actually resolving a face.
"""
import numpy as np
import pytest

from core.active_challenge import ActiveChallenge
from utils.constants import (
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    NOSE_TIP_IDX,
    HEAD_TURN_THRESHOLD,
)

_DUMMY_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


def _landmarks_with_nose_at(nose_x):
    """468 landmarks, all at the origin except the cheeks (fixed apart) and
    the nose tip (placed at `nose_x`) - everything evaluate_single_frame /
    run_challenge actually reads for yaw."""
    landmarks = [(0.0, 0.0)] * 468
    landmarks[LEFT_CHEEK_IDX] = (0.0, 0.0)
    landmarks[RIGHT_CHEEK_IDX] = (100.0, 0.0)
    landmarks[NOSE_TIP_IDX] = (nose_x, 0.0)
    return landmarks


# cheek_mid_x = 50, cheek_dist = 100 -> yaw_offset = (nose_x - 50) / 100
_NOSE_X_POSITIVE_YAW = 50.0 + (HEAD_TURN_THRESHOLD + 0.05) * 100.0
_NOSE_X_NEGATIVE_YAW = 50.0 - (HEAD_TURN_THRESHOLD + 0.05) * 100.0


@pytest.fixture
def active_challenge(monkeypatch):
    ac = ActiveChallenge()
    yield ac
    ac.close()


def test_positive_yaw_offset_passes_turn_right_not_turn_left(active_challenge, monkeypatch):
    # Sign corrected 2026-08-21 against real demo testing: a physical RIGHT
    # turn measures as positive yaw_offset (previously had this backwards -
    # see the required_sign comment in evaluate_single_frame()).
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: _landmarks_with_nose_at(_NOSE_X_POSITIVE_YAW),
    )

    left = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_left")
    right = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_right")

    assert left["passed"] is False
    assert right["passed"] is True


def test_negative_yaw_offset_passes_turn_left_not_turn_right(active_challenge, monkeypatch):
    # Sign corrected 2026-08-21 - see test_positive_yaw_offset_... above.
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: _landmarks_with_nose_at(_NOSE_X_NEGATIVE_YAW),
    )

    left = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_left")
    right = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_right")

    assert left["passed"] is True
    assert right["passed"] is False


class _FakeCap:
    """Stands in for cv2.VideoCapture: always returns the same frame."""
    def read(self):
        return True, _DUMMY_FRAME


def test_run_challenge_turn_left_and_turn_right_are_self_consistent_opposites(
    active_challenge, monkeypatch,
):
    # run_challenge() (the --live path, raw cv2.VideoCapture frames) already
    # used the same sign convention evaluate_single_frame() was just
    # corrected to match (2026-08-21, confirmed against real demo testing) -
    # this pins that self-consistency (turn_left/turn_right can't both fire
    # for the same yaw) and cross-checks it against evaluate_single_frame
    # for the same landmarks, so the two paths can't silently diverge again.
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: _landmarks_with_nose_at(_NOSE_X_NEGATIVE_YAW),
    )

    right = active_challenge.run_challenge(
        _FakeCap(), challenge="turn_right", show_window=False, timeout=0.5,
    )
    left = active_challenge.run_challenge(
        _FakeCap(), challenge="turn_left", show_window=False, timeout=0.2,
    )
    assert right["passed"] != left["passed"]

    single_frame_left = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_left")
    single_frame_right = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_right")
    assert left["passed"] == single_frame_left["passed"]
    assert right["passed"] == single_frame_right["passed"]
