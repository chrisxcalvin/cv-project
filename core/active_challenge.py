# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Active liveness challenge: blink / turn-left / turn-right, verified via
MediaPipe Face Mesh landmarks.

SYLLABUS: EAR / Landmark tracking - Eye Aspect Ratio (Soukupova & Cech,
2016) computed from six eye-contour landmarks per eye, used to detect
blinks with N-frame confirmation to reject single-frame noise.

SYLLABUS: Head pose estimation - yaw is approximated cheaply from the
nose-tip offset relative to the cheek midpoint (no full 3D solvePnP needed
for a binary "turned enough" decision).
"""
import time
import random

import cv2
import numpy as np
import mediapipe as mp

from utils.constants import (
    LEFT_EYE_EAR_IDX,
    RIGHT_EYE_EAR_IDX,
    LEFT_EYE_RING_IDX,
    RIGHT_EYE_RING_IDX,
    NOSE_TIP_IDX,
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    MOUTH_MAR_IDX,
    EAR_BLINK_THRESHOLD,
    EAR_BLINK_THRESHOLD_GLASSES,
    HEAD_TURN_THRESHOLD,
    MOUTH_OPEN_THRESHOLD,
    N_FRAME_CONFIRM,
    CHALLENGE_TIMEOUT,
)

mp_face_mesh = mp.solutions.face_mesh

CHALLENGES = ["blink", "turn_left", "turn_right", "mouth_open"]


def _euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _ear_from_points(pts):
    """pts = [p1, p2, p3, p4, p5, p6] in the standard EAR ordering."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def _mar_from_points(pts):
    """
    pts = [left_corner, upper_inner_lip, right_corner, lower_inner_lip].

    SYLLABUS: EAR / Landmark tracking - Mouth Aspect Ratio, the same
    vertical/horizontal-distance-ratio idea as the Eye Aspect Ratio above,
    applied to the mouth to detect an open-mouth / smile-with-teeth gesture.
    """
    left, top, right, bottom = pts
    vertical = _euclidean(top, bottom)
    horizontal = _euclidean(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


class ActiveChallenge:
    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------
    def _get_landmarks(self, frame):
        """Return a list of (x, y) pixel coordinates for all 468 landmarks, or None."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        return [(lm.x * w, lm.y * h) for lm in landmarks]

    def _compute_ear(self, landmarks):
        left_pts = [landmarks[i] for i in LEFT_EYE_EAR_IDX]
        right_pts = [landmarks[i] for i in RIGHT_EYE_EAR_IDX]
        left_ear = _ear_from_points(left_pts)
        right_ear = _ear_from_points(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        return left_ear, right_ear, avg_ear

    def _compute_mar(self, landmarks):
        pts = [landmarks[i] for i in MOUTH_MAR_IDX]
        return _mar_from_points(pts)

    def _compute_yaw(self, landmarks):
        """
        Cheap yaw proxy: horizontal offset of the nose tip from the cheek
        midpoint, normalised by inter-cheek distance.

        yaw_offset > 0  -> nose shifted toward the RIGHT_CHEEK_IDX side
        yaw_offset < 0  -> nose shifted toward the LEFT_CHEEK_IDX side

        NOTE: browser webcam captures (the browser-camera path used by
        webapp.py, and cv2.VideoCapture on a laptop webcam) are effectively
        mirrored relative to the anatomical labelling of MediaPipe's
        canonical face mesh - a user physically turning their head to their
        own right measures as yaw_offset > 0 here, turning left as
        yaw_offset < 0 (confirmed empirically 2026-08-21). Both
        evaluate_single_frame() and run_challenge() use this same
        convention - see evaluate_single_frame()'s required_sign for the
        turn_left/turn_right mapping.
        """
        nose_x, _ = landmarks[NOSE_TIP_IDX]
        left_x, _ = landmarks[LEFT_CHEEK_IDX]
        right_x, _ = landmarks[RIGHT_CHEEK_IDX]

        cheek_mid_x = (left_x + right_x) / 2.0
        cheek_dist = abs(right_x - left_x)
        if cheek_dist < 1e-6:
            return 0.0
        return (nose_x - cheek_mid_x) / cheek_dist

    def pick_random_challenge(self):
        return random.choice(CHALLENGES)

    def get_ear(self, frame):
        """Return the average Eye Aspect Ratio for `frame`, or None if no face is found."""
        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return None
        _, _, avg_ear = self._compute_ear(landmarks)
        return avg_ear

    def get_mar(self, frame):
        """Return the Mouth Aspect Ratio for `frame`, or None if no face is found."""
        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return None
        return self._compute_mar(landmarks)

    # ------------------------------------------------------------------
    # Single-frame evaluation (used by the Streamlit dashboard, which can
    # only capture discrete photos via st.camera_input, not a live stream)
    # ------------------------------------------------------------------
    def evaluate_single_frame(self, frame, challenge, glasses_mode=False):
        """
        Evaluate a single captured frame against `challenge`.
        Returns a result dict: passed / challenge / reason / ear_min / yaw_max_delta.
        """
        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return {
                "passed": False, "challenge": challenge,
                "reason": "no_face_detected", "ear_min": None, "yaw_max_delta": None,
                "yaw_signed": None,
            }

        ear_threshold = EAR_BLINK_THRESHOLD_GLASSES if glasses_mode else EAR_BLINK_THRESHOLD
        _, _, avg_ear = self._compute_ear(landmarks)
        yaw_offset = self._compute_yaw(landmarks)

        if challenge == "blink":
            passed = avg_ear < ear_threshold
            reason = "eyes_closed_detected" if passed else "eyes_open_no_blink"
            return {
                "passed": passed, "challenge": challenge, "reason": reason,
                "ear_min": round(avg_ear, 4), "yaw_max_delta": round(abs(yaw_offset), 4),
                "yaw_signed": round(yaw_offset, 4),
            }

        if challenge in ("turn_left", "turn_right"):
            # Sign convention confirmed empirically against the real
            # browser capture pipeline during demo testing (2026-08-21):
            # physically turning left produces a NEGATIVE yaw_offset here,
            # turning right a POSITIVE one - matching run_challenge()'s
            # (the --live/cv2.VideoCapture path) convention below, which
            # this function previously had backwards relative to.
            required_sign = -1 if challenge == "turn_left" else 1
            passed = (yaw_offset * required_sign) >= HEAD_TURN_THRESHOLD
            reason = "head_turn_detected" if passed else "head_turn_insufficient"
            return {
                "passed": passed, "challenge": challenge, "reason": reason,
                "ear_min": round(avg_ear, 4), "yaw_max_delta": round(abs(yaw_offset), 4),
                "yaw_signed": round(yaw_offset, 4),
            }

        if challenge == "mouth_open":
            mar = self._compute_mar(landmarks)
            passed = mar > MOUTH_OPEN_THRESHOLD
            reason = "mouth_open_detected" if passed else "mouth_still_closed"
            return {
                "passed": passed, "challenge": challenge, "reason": reason,
                "ear_min": round(avg_ear, 4), "yaw_max_delta": round(abs(yaw_offset), 4),
                "yaw_signed": round(yaw_offset, 4), "mar": round(mar, 4),
            }

        return {
            "passed": False, "challenge": challenge,
            "reason": "unknown_challenge_type", "ear_min": None, "yaw_max_delta": None,
        }

    # ------------------------------------------------------------------
    # Burst evaluation (used by the Flask web app - webapp.py sends a short
    # burst of frames captured over ~1s rather than one photo)
    # ------------------------------------------------------------------
    def evaluate_burst(self, frames, challenge, glasses_mode=False):
        """
        Evaluate an ordered burst of frames against `challenge`, requiring
        N_FRAME_CONFIRM CONSECUTIVE frames to satisfy the gesture - the same
        confirmation rule run_challenge() already uses for --live mode.

        Demo testing found that accepting the FIRST single frame that
        passes (the old per-frame loop in webapp.py) lets someone fake a
        turn_left/turn_right by briefly tilting a printed/phone photo in
        front of the camera: MediaPipe reads the resulting 2D landmark
        shift the same as a real head turn, and only needs one lucky
        frame. Requiring a consecutive streak makes that much harder to
        fake with a quick tilt or a single well-angled video frame, and
        gives fusion.py's anti-swap check a whole streak of frames to
        re-verify instead of just one (see webapp.py's /api/challenge).

        Returns (result, confirmed_frames):
          result           - same shape as evaluate_single_frame()'s return
          confirmed_frames - the list of raw frames making up the winning
                             streak (empty if the challenge was never
                             confirmed)
        """
        streak = []
        best_attempt = None

        for frame in frames:
            result = self.evaluate_single_frame(frame, challenge, glasses_mode=glasses_mode)

            streak = streak + [(frame, result)] if result["passed"] else []

            if best_attempt is None or (
                result["reason"] != "no_face_detected"
                and best_attempt["reason"] == "no_face_detected"
            ):
                best_attempt = result

            if len(streak) >= N_FRAME_CONFIRM:
                confirmed_frames = [f for f, _ in streak]
                final_result = dict(streak[-1][1])
                final_result["reason"] = f"{challenge}_confirmed_{N_FRAME_CONFIRM}_frames"
                return final_result, confirmed_frames

        # Never confirmed a full streak. `best_attempt` is kept only for a
        # useful diagnostic (it prefers a face-tracked frame over a
        # no-face one) - it must NOT be allowed to report passed=True on
        # its own, since it can legitimately BE the one lucky frame this
        # method exists to stop from deciding the outcome by itself.
        fallback = dict(best_attempt) if best_attempt is not None else {
            "passed": False, "challenge": challenge, "reason": "no_face_detected",
            "ear_min": None, "yaw_max_delta": None, "yaw_signed": None,
        }
        if fallback["passed"]:
            fallback["reason"] = f"{fallback['reason']}_not_sustained"
        fallback["passed"] = False
        return fallback, []

    # ------------------------------------------------------------------
    # Continuous multi-frame challenge (used by main.py --live mode)
    # ------------------------------------------------------------------
    def run_challenge(self, cap, challenge=None, glasses_mode=False,
                       show_window=True, timeout=CHALLENGE_TIMEOUT):
        """
        Run a live active-challenge loop reading frames from an already-open
        cv2.VideoCapture `cap`, until the challenge passes, fails, or times out.

        SYLLABUS: N-frame confirmation - a blink/turn must persist for
        N_FRAME_CONFIRM consecutive frames before it counts, to reject
        single-frame detector noise.

        Returns a result dict: passed / challenge / reason / ear_min / yaw_max_delta.
        """
        if challenge is None:
            challenge = self.pick_random_challenge()

        ear_threshold = EAR_BLINK_THRESHOLD_GLASSES if glasses_mode else EAR_BLINK_THRESHOLD
        start_time = time.time()

        confirm_streak = 0
        ear_min = 1.0
        yaw_max_delta = 0.0
        passed = False
        reason = "timeout"

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            remaining = timeout - (time.time() - start_time)
            landmarks = self._get_landmarks(frame)

            if landmarks is None:
                confirm_streak = 0
                if show_window:
                    self._draw_challenge_frame(frame, challenge, remaining, None, None)
                    cv2.imshow("DeepCheck - Active Challenge", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        reason = "cancelled"
                        break
                continue

            _, _, avg_ear = self._compute_ear(landmarks)
            yaw_offset = self._compute_yaw(landmarks)
            mar = self._compute_mar(landmarks)
            ear_min = min(ear_min, avg_ear)
            yaw_max_delta = max(yaw_max_delta, abs(yaw_offset))

            # NOTE: this reads raw cv2.VideoCapture frames, a different
            # capture pipeline from the Streamlit st.camera_input path in
            # evaluate_single_frame() above (which needed its turn_left/
            # turn_right sign flipped to match browser mirroring). If
            # --live mode's turn detection also reads backwards for you,
            # flip these two comparisons the same way.
            if challenge == "blink":
                frame_ok = avg_ear < ear_threshold
            elif challenge == "turn_left":
                frame_ok = yaw_offset <= -HEAD_TURN_THRESHOLD
            elif challenge == "turn_right":
                frame_ok = yaw_offset >= HEAD_TURN_THRESHOLD
            elif challenge == "mouth_open":
                frame_ok = mar > MOUTH_OPEN_THRESHOLD
            else:
                frame_ok = False

            confirm_streak = confirm_streak + 1 if frame_ok else 0

            if show_window:
                self._draw_challenge_frame(frame, challenge, remaining, avg_ear, yaw_offset)
                cv2.imshow("DeepCheck - Active Challenge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    reason = "cancelled"
                    break

            if confirm_streak >= N_FRAME_CONFIRM:
                passed = True
                reason = f"{challenge}_confirmed_{N_FRAME_CONFIRM}_frames"
                break

        if show_window:
            cv2.destroyWindow("DeepCheck - Active Challenge")

        if not passed and reason == "timeout":
            reason = f"{challenge}_timeout"

        return {
            "passed": passed,
            "challenge": challenge,
            "reason": reason,
            "ear_min": round(float(ear_min), 4),
            "yaw_max_delta": round(float(yaw_max_delta), 4),
        }

    def _draw_challenge_frame(self, frame, challenge, remaining, avg_ear, yaw_offset):
        prompt = {
            "blink": "Please BLINK",
            "turn_left": "Please TURN LEFT",
            "turn_right": "Please TURN RIGHT",
            "mouth_open": "Please OPEN YOUR MOUTH",
        }.get(challenge, challenge)

        cv2.putText(frame, prompt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"time left: {remaining:0.1f}s", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if avg_ear is not None:
            cv2.putText(frame, f"EAR: {avg_ear:0.3f}", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if yaw_offset is not None:
            cv2.putText(frame, f"yaw: {yaw_offset:0.3f}", (30, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self):
        self._mesh.close()
