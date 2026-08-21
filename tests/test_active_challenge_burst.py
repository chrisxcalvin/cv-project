"""
Pins ActiveChallenge.evaluate_burst()'s consecutive-frame confirmation rule,
added after demo testing showed a single lucky frame (e.g. a printed/phone
photo briefly tilted to fake a head turn) was enough to pass a challenge
under the old per-frame-in-a-loop logic in webapp.py's /api/challenge.

Landmarks are faked via monkeypatching ActiveChallenge._get_landmarks, so
these tests don't depend on mediapipe actually resolving a face - same
pattern as tests/test_active_challenge_yaw_sign.py.
"""
import numpy as np
import pytest

from core.active_challenge import ActiveChallenge
from utils.constants import (
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    NOSE_TIP_IDX,
    HEAD_TURN_THRESHOLD,
    N_FRAME_CONFIRM,
)

# Negative offset: turn_left passes on NEGATIVE yaw_offset (sign
# convention corrected 2026-08-21 - see the required_sign comment in
# ActiveChallenge.evaluate_single_frame()). All tests in this file
# evaluate against "turn_left" specifically.
_PASSING_YAW_NOSE_X = 50.0 - (HEAD_TURN_THRESHOLD + 0.05) * 100.0
_FAILING_YAW_NOSE_X = 50.0  # centered - no turn at all


def _landmarks(nose_x):
    landmarks = [(0.0, 0.0)] * 468
    landmarks[LEFT_CHEEK_IDX] = (0.0, 0.0)
    landmarks[RIGHT_CHEEK_IDX] = (100.0, 0.0)
    landmarks[NOSE_TIP_IDX] = (nose_x, 0.0)
    return landmarks


_PASSING_LANDMARKS = _landmarks(_PASSING_YAW_NOSE_X)
_FAILING_LANDMARKS = _landmarks(_FAILING_YAW_NOSE_X)


@pytest.fixture
def active_challenge():
    ac = ActiveChallenge()
    yield ac
    ac.close()


def _patch_frames(monkeypatch, active_challenge, passing_flags):
    """Build one dummy frame per flag in `passing_flags` and make
    _get_landmarks return passing/failing landmarks per-frame based on
    frame identity (not real pixel content)."""
    frames = [np.array([i]) for i in range(len(passing_flags))]
    landmark_by_id = {
        id(frame): (_PASSING_LANDMARKS if passed else _FAILING_LANDMARKS)
        for frame, passed in zip(frames, passing_flags)
    }
    monkeypatch.setattr(
        active_challenge, "_get_landmarks",
        lambda frame: landmark_by_id[id(frame)],
    )
    return frames


def test_single_passing_frame_among_many_does_not_confirm(active_challenge, monkeypatch):
    # Exactly one lucky frame (e.g. a tilted photo snapping past the yaw
    # threshold for an instant) must NOT be enough to pass the challenge.
    flags = [False] * 5 + [True] + [False] * 6
    frames = _patch_frames(monkeypatch, active_challenge, flags)

    result, confirmed = active_challenge.evaluate_burst(frames, "turn_left")

    assert result["passed"] is False
    assert confirmed == []


def test_n_consecutive_passing_frames_confirms(active_challenge, monkeypatch):
    flags = [False] * 4 + [True] * N_FRAME_CONFIRM + [False] * 3
    frames = _patch_frames(monkeypatch, active_challenge, flags)

    result, confirmed = active_challenge.evaluate_burst(frames, "turn_left")

    assert result["passed"] is True
    assert len(confirmed) == N_FRAME_CONFIRM
    assert result["reason"] == f"turn_left_confirmed_{N_FRAME_CONFIRM}_frames"


def test_streak_resets_on_an_interrupting_failing_frame(active_challenge, monkeypatch):
    # Two separate short passing runs, each shorter than N_FRAME_CONFIRM,
    # broken up by a failing frame - must not confirm even though the
    # total passing-frame count across the whole burst reaches N_FRAME_CONFIRM.
    assert N_FRAME_CONFIRM >= 2, "test assumes N_FRAME_CONFIRM >= 2"
    half = N_FRAME_CONFIRM - 1
    flags = [True] * half + [False] + [True] * half
    frames = _patch_frames(monkeypatch, active_challenge, flags)

    result, confirmed = active_challenge.evaluate_burst(frames, "turn_left")

    assert result["passed"] is False
    assert confirmed == []


def test_empty_burst_reports_no_face_detected(active_challenge):
    result, confirmed = active_challenge.evaluate_burst([], "turn_left")

    assert result["passed"] is False
    assert result["reason"] == "no_face_detected"
    assert confirmed == []
