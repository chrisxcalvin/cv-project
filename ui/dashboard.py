# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Streamlit demo dashboard for DeepCheck.

CRITICAL: uses st.camera_input (photo capture) rather than a raw OpenCV
VideoCapture loop, because OpenCV webcam access does not work reliably
inside a Streamlit app process.
"""
import io
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_detector import FaceDetector
from core.passive_classifier import PassiveClassifier
from core.active_challenge import ActiveChallenge
from core.fusion import DecisionFusion
from utils import logger as db_logger
from utils.constants import EAR_BLINK_THRESHOLD, HEAD_TURN_THRESHOLD, MOUTH_OPEN_THRESHOLD, MAX_FACES_ALLOWED

st.set_page_config(page_title="DeepCheck - Liveness Detection", layout="wide")


@st.cache_resource
def get_face_detector():
    return FaceDetector()


@st.cache_resource
def get_passive_classifier():
    return PassiveClassifier()


@st.cache_resource
def get_active_challenge():
    return ActiveChallenge()


@st.cache_resource
def get_decision_fusion():
    return DecisionFusion()


def camera_input_to_bgr(img_file):
    """Convert a st.camera_input UploadedFile into an OpenCV BGR ndarray."""
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def build_verdict_report_png(face_crop, passive_result, decision) -> bytes:
    """Compose a one-page PNG summary (face crop + feature views + scores +
    verdict) so a demo attempt can be saved/shared instead of only living in
    the browser session."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    axes[0].imshow(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Face crop")
    axes[1].imshow(passive_result["edges"], cmap="gray")
    axes[1].set_title("Canny edges")
    axes[2].imshow(cv2.cvtColor(passive_result["kp_frame"], cv2.COLOR_BGR2RGB))
    axes[2].set_title("ORB keypoints")
    axes[3].imshow(cv2.cvtColor(passive_result["fft_display"], cv2.COLOR_BGR2RGB))
    axes[3].set_title("FFT spectrum")
    for ax in axes:
        ax.axis("off")

    summary_lines = [
        f"Verdict: {decision['verdict']}  (confidence: {decision['confidence'] * 100:0.1f}%)",
        f"Method: {passive_result['method']}"
        + (f"  |  Spoof type: {passive_result['spoof_type']}" if passive_result.get("spoof_type") else ""),
        f"Texture var: {passive_result['texture_var']}   ORB kp: {passive_result['orb_keypoints']}   "
        f"FFT ratio: {passive_result['fft_ratio']}   Edge density: {passive_result['edge_density']}",
    ]
    fig.suptitle("\n".join(summary_lines), fontsize=9, y=1.08, ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_live_demo_tab():
    st.header("Live Demo")
    st.caption("Capture a photo below to run the full DeepCheck liveness pipeline.")

    face_detector = get_face_detector()
    classifier = get_passive_classifier()
    active_challenge = get_active_challenge()
    fusion = get_decision_fusion()

    if "pending_challenge" not in st.session_state:
        st.session_state.pending_challenge = None
    if "pending_passive_result" not in st.session_state:
        st.session_state.pending_passive_result = None
    if "last_main_capture_id" not in st.session_state:
        st.session_state.last_main_capture_id = None
    if "confidence_history" not in st.session_state:
        st.session_state.confidence_history = []
    if "challenge_retry_count" not in st.session_state:
        st.session_state.challenge_retry_count = 0

    img_file = st.camera_input("Capture your face", key="main_capture")

    if img_file is not None:
        # A fresh main-photo capture must start a fresh challenge cycle -
        # otherwise a leftover pending_challenge/pending_passive_result from
        # a previous, never-finished attempt gets silently reused below.
        if img_file.file_id != st.session_state.last_main_capture_id:
            st.session_state.pending_challenge = None
            st.session_state.pending_passive_result = None
            st.session_state.challenge_retry_count = 0
            st.session_state.last_main_capture_id = img_file.file_id

        frame = camera_input_to_bgr(img_file)
        face_crop, bbox = face_detector.detect_and_crop(frame)
        num_faces = face_detector.count_faces(frame)

        if face_crop is None:
            st.error("No face detected. Please recapture with your face clearly visible.")
            st.session_state.pending_challenge = None
            st.session_state.pending_passive_result = None
            st.session_state.challenge_retry_count = 0
        elif num_faces > MAX_FACES_ALLOWED:
            # Blocks a common attack: holding a printed photo up next to
            # your own real face so either one can be swapped in as needed.
            st.error(
                f"{num_faces} faces detected - only one person may attempt "
                "verification at a time. Please recapture with just yourself in frame."
            )
            st.session_state.pending_challenge = None
            st.session_state.pending_passive_result = None
            st.session_state.challenge_retry_count = 0
        else:
            passive_result = classifier.predict(face_crop)
            decision = fusion.decide(passive_result)
            st.session_state.confidence_history.append(passive_result["real_confidence"])

            st.subheader("Real-Confidence Score")
            st.progress(min(max(passive_result["real_confidence"], 0.0), 1.0))
            st.metric("Real confidence", f"{passive_result['real_confidence'] * 100:0.1f}%")
            if passive_result.get("spoof_type"):
                st.caption(f"Likely spoof type: **{passive_result['spoof_type'].replace('_', ' ')}**")

            st.subheader("Feature Views")
            cols = st.columns(4)
            with cols[0]:
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="Face crop")
            with cols[1]:
                st.image(passive_result["edges"], caption="Canny edges", clamp=True)
            with cols[2]:
                st.image(cv2.cvtColor(passive_result["kp_frame"], cv2.COLOR_BGR2RGB),
                          caption="ORB keypoints")
            with cols[3]:
                st.image(cv2.cvtColor(passive_result["fft_display"], cv2.COLOR_BGR2RGB),
                          caption="FFT spectrum")

            st.subheader("Feature Scores")
            st.table({
                "Feature": ["Texture variance (Otsu)", "ORB keypoints",
                            "FFT high-freq ratio", "Canny edge density", "Method"],
                "Value": [passive_result["texture_var"], passive_result["orb_keypoints"],
                          passive_result["fft_ratio"], passive_result["edge_density"],
                          passive_result["method"]],
            })

            if passive_result.get("weighted_breakdown"):
                st.subheader("Explainability: Weighted Contribution Breakdown")
                st.caption(
                    "How many of the final real-confidence points each feature "
                    "actually contributed (weight x normalised sub-score), not "
                    "just its raw value above."
                )
                breakdown = passive_result["weighted_breakdown"]
                st.bar_chart({
                    "contribution": {
                        "Texture (35%)": breakdown["texture"],
                        "ORB (25%)": breakdown["orb"],
                        "FFT (25%)": breakdown["fft"],
                        "Edge (15%)": breakdown["edge"],
                    }
                })

            if len(st.session_state.confidence_history) > 1:
                st.subheader("Session Confidence Trend")
                st.caption("Real-confidence across every capture this session.")
                st.line_chart(st.session_state.confidence_history)

            active_result = None
            if decision["verdict"] == "PENDING":
                if st.session_state.pending_challenge is None:
                    st.session_state.pending_challenge = active_challenge.pick_random_challenge()
                    st.session_state.pending_passive_result = passive_result

                challenge = st.session_state.pending_challenge
                prompt = {
                    "blink": "Blink your eyes, then capture the photo below.",
                    "turn_left": "Turn your head slightly LEFT - just a few degrees, "
                                 "keep most of your face visible - then capture the photo below.",
                    "turn_right": "Turn your head slightly RIGHT - just a few degrees, "
                                  "keep most of your face visible - then capture the photo below.",
                    "mouth_open": "Open your mouth, then capture the photo below.",
                }[challenge]

                st.warning(f"Challenge Required: {challenge.upper()}")
                st.info(prompt)

                challenge_file = st.camera_input("Capture challenge photo", key="challenge_capture")
                if challenge_file is not None:
                    challenge_frame = camera_input_to_bgr(challenge_file)
                    challenge_crop, _ = face_detector.detect_and_crop(challenge_frame)
                    if challenge_crop is None:
                        st.error("No face detected in challenge photo.")
                    else:
                        # Run EAR/yaw landmark geometry on the original,
                        # undistorted frame - not the square-resized crop.
                        # detect_and_crop() force-resizes to a fixed square
                        # (see face_detector.py), which warps the vertical vs.
                        # horizontal proportions that EAR and yaw depend on.
                        active_result = active_challenge.evaluate_single_frame(
                            challenge_frame, challenge
                        )

                        if (active_result["reason"] == "no_face_detected"
                                and st.session_state.challenge_retry_count < 1):
                            # Face Mesh needs more of the face's geometry
                            # visible than the plain face detector above, so
                            # it can fail to resolve landmarks (e.g. an
                            # over-turned head) even when a face was clearly
                            # present. That's usually a one-off tracking
                            # hiccup for a genuine user, so give ONE free
                            # retry instead of instantly wiping the whole
                            # attempt (decision stays PENDING from the
                            # passive-only check above, so nothing is logged
                            # yet). Capped at one retry, though: a flat
                            # printed/screen photo tilted to fake a turn
                            # will keep failing to track no matter how many
                            # times it's retried, so unlimited retries would
                            # just let that kind of spoof attempt keep
                            # rolling the dice instead of being rejected.
                            st.session_state.challenge_retry_count += 1
                            st.warning(
                                "Couldn't track facial landmarks on that photo "
                                "(often happens if the turn was too far, or the "
                                "face was partly out of frame). Keep more of "
                                "your face visible and recapture the challenge "
                                "photo below."
                            )
                        else:
                            # The challenge photo must ALSO not look like a spoof -
                            # otherwise a fake photo could be shown first (landing
                            # in the uncertain band) and then swapped for a real
                            # face just to satisfy the gesture.
                            challenge_passive_result = classifier.predict(challenge_crop)
                            decision = fusion.decide(
                                st.session_state.pending_passive_result, active_result,
                                challenge_passive_result,
                            )

                            st.subheader("Challenge Measurement")
                            st.write(
                                f"Challenge photo spoof-check: real_confidence = "
                                f"{challenge_passive_result['real_confidence']:.2f} "
                                f"(must be above {DecisionFusion.REJECT_THRESHOLD:.2f})"
                            )
                            if challenge == "blink":
                                st.write(f"EAR must drop below {EAR_BLINK_THRESHOLD:.2f} (yours: {active_result['ear_min']:.3f})")
                            elif challenge == "turn_left":
                                st.write(f"signed yaw offset must be ≥ {HEAD_TURN_THRESHOLD:.2f} (yours: {active_result['yaw_signed']:.3f})")
                            elif challenge == "turn_right":
                                st.write(f"signed yaw offset must be ≤ -{HEAD_TURN_THRESHOLD:.2f} (yours: {active_result['yaw_signed']:.3f})")
                            elif challenge == "mouth_open":
                                st.write(f"MAR must exceed {MOUTH_OPEN_THRESHOLD:.2f} (yours: {active_result['mar']:.3f})")
                            st.write(f"Result: **{'PASSED' if active_result['passed'] else 'FAILED'}** ({active_result['reason']})")

                            st.session_state.pending_challenge = None
                            st.session_state.pending_passive_result = None
                            st.session_state.challenge_retry_count = 0
            else:
                st.session_state.pending_challenge = None
                st.session_state.pending_passive_result = None
                st.session_state.challenge_retry_count = 0

            if decision["verdict"] != "PENDING":
                st.subheader("Final Verdict")
                if decision["verdict"] == "ACCEPT":
                    st.success(f"ACCEPT  (confidence: {decision['confidence'] * 100:0.1f}%)")
                else:
                    st.error(f"REJECT  (confidence: {decision['confidence'] * 100:0.1f}%)")

                db_logger.log_attempt(passive_result, decision, active_result)

                report_png = build_verdict_report_png(face_crop, passive_result, decision)
                st.download_button(
                    "Download verdict report (PNG)", data=report_png,
                    file_name="deepcheck_report.png", mime="image/png",
                )


def render_analytics_tab():
    st.header("Analytics")

    stats = db_logger.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total checks", stats["total"])
    col2.metric("Accept rate", f"{stats['accept_rate'] * 100:0.1f}%")
    col3.metric("Reject rate", f"{stats['reject_rate'] * 100:0.1f}%")
    col4.metric("Challenge trigger rate", f"{stats['challenge_rate'] * 100:0.1f}%")

    recent = db_logger.get_recent(20)
    if recent:
        st.subheader("Verdicts (last 20 attempts)")
        verdict_counts = {"ACCEPT": 0, "REJECT": 0, "PENDING": 0}
        for row in recent:
            verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
        st.bar_chart(verdict_counts)

        st.subheader("Last 10 Attempts")
        last_10 = recent[:10]
        st.dataframe([
            {
                "timestamp": r["timestamp"],
                "verdict": r["verdict"],
                "real_confidence": r["real_confidence"],
                "texture_var": r["texture_var"],
                "orb_keypoints": r["orb_keypoints"],
                "fft_ratio": r["fft_ratio"],
                "edge_density": r["edge_density"],
                "method": r["method"],
                "spoof_type": r["spoof_type"],
                "active_used": bool(r["active_used"]),
                "challenge_type": r["challenge_type"],
                "reason": r["reason"],
                "ear_min": r["ear_min"],
                "yaw_max_delta": r["yaw_max_delta"],
            }
            for r in last_10
        ])
    else:
        st.info("No attempts logged yet. Run the Live Demo tab to generate data.")


def main():
    st.title("DeepCheck")
    st.caption("Real-time face liveness and anti-spoofing detection demo")

    tab1, tab2 = st.tabs(["Live Demo", "Analytics"])
    with tab1:
        render_live_demo_tab()
    with tab2:
        render_analytics_tab()


if __name__ == "__main__":
    main()
