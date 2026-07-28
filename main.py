# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
DeepCheck entry point.

Mode 1 (default) - Streamlit dashboard:
    python main.py

Mode 2 - OpenCV live window:
    python main.py --live
"""
import argparse
import subprocess
import sys
import os
from collections import deque

import cv2

from core.face_detector import FaceDetector
from core.passive_classifier import PassiveClassifier
from core.active_challenge import ActiveChallenge
from core.fusion import DecisionFusion
from ui.visualizer import (
    draw_verdict_overlay,
    draw_feature_panel,
    draw_ear_graph,
    draw_challenge_prompt,
    draw_fft_inset,
)
from utils import logger as db_logger
from utils.constants import EAR_HISTORY_LEN


def run_streamlit():
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def run_live():
    db_logger.init_db()

    face_detector = FaceDetector()
    classifier = PassiveClassifier()
    active_challenge = ActiveChallenge()
    fusion = DecisionFusion()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[main] Could not open webcam.")
        return

    ear_history = deque(maxlen=EAR_HISTORY_LEN)
    print("[main] Live mode running. Press SPACE to trigger an active challenge, Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            face_crop, bbox = face_detector.detect_and_crop(frame)
            display = frame.copy()

            if face_crop is None:
                cv2.putText(display, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                face_detector.draw_bbox(display, bbox)

                passive_result = classifier.predict(face_crop)
                decision = fusion.decide(passive_result)

                avg_ear = active_challenge.get_ear(face_crop)
                if avg_ear is not None:
                    ear_history.append(avg_ear)

                draw_verdict_overlay(display, decision["verdict"], decision["confidence"])
                draw_feature_panel(display, passive_result)
                draw_ear_graph(display, ear_history, 220, 100)
                draw_fft_inset(display, passive_result["fft_display"])

            cv2.imshow("DeepCheck - Live", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" ") and face_crop is not None:
                challenge = active_challenge.pick_random_challenge()
                print(f"[main] Running active challenge: {challenge}")
                active_result = active_challenge.run_challenge(
                    cap, challenge=challenge, show_window=True
                )
                print(f"[main] Challenge result: {active_result}")

                final_passive = classifier.predict(face_crop)
                final_decision = fusion.decide(final_passive, active_result)
                db_logger.log_attempt(final_passive, final_decision, active_result)
                print(f"[main] Final verdict: {final_decision['verdict']}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()
        active_challenge.close()


def main():
    parser = argparse.ArgumentParser(description="DeepCheck - face liveness detection")
    parser.add_argument("--live", action="store_true",
                         help="Run the OpenCV live webcam window instead of the Streamlit dashboard.")
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_streamlit()


if __name__ == "__main__":
    main()
