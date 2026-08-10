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


def test_positive_yaw_offset_passes_turn_left_not_turn_right(active_challenge, monkeypatch):
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: _landmarks_with_nose_at(_NOSE_X_POSITIVE_YAW),
    )

    left = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_left")
    right = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_right")

    assert left["passed"] is True
    assert right["passed"] is False


def test_negative_yaw_offset_passes_turn_right_not_turn_left(active_challenge, monkeypatch):
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: _landmarks_with_nose_at(_NOSE_X_NEGATIVE_YAW),
    )

    left = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_left")
    right = active_challenge.evaluate_single_frame(_DUMMY_FRAME, "turn_right")

    assert left["passed"] is False
    assert right["passed"] is True


class _FakeCap:
    """Stands in for cv2.VideoCapture: always returns the same frame."""
    def read(self):
        return True, _DUMMY_FRAME


def test_run_challenge_turn_left_and_turn_right_are_self_consistent_opposites(
    active_challenge, monkeypatch,
):
    # NOTE: run_challenge() (the --live path, raw cv2.VideoCapture frames) is
    # NOT asserted here to share evaluate_single_frame()'s sign convention
    # (the Streamlit st.camera_input path) - browser-captured photos and raw
    # webcam frames can have different mirroring depending on OS/driver, and
    # core/active_challenge.py's own comments flag this as unresolved
    # ("if --live mode's turn detection also reads backwards for you, flip
    # these two comparisons the same way"). Verify the actual direction
    # against a real webcam in --live mode before demo day; this test only
    # pins that turn_left and turn_right can't both fire for the same yaw.
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
