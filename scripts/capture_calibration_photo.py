"""
Captures webcam photos into calibration_data/<category>/ for threshold
calibration (see calibration_data/README.md).

Uses the SAME webcam + face-detect-and-crop path as the live app
(core/face_detector.py) so the calibration numbers measure exactly what
PassiveClassifier.predict() will see in the real demo - not a phone photo
run through a different camera/lens/compression pipeline.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\capture_calibration_photo.py --name vansh --category real
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_detector import FaceDetector

CATEGORIES = ("real", "print", "replay")
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_data")


def next_index(out_dir, name, category):
    existing = glob.glob(os.path.join(out_dir, f"{name}_{category}_*.jpg"))
    return len(existing) + 1


def main():
    parser = argparse.ArgumentParser(description="Capture calibration photos.")
    parser.add_argument("--name", required=True, help="Your name, used as filename prefix.")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.category)
    os.makedirs(out_dir, exist_ok=True)

    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[capture] Could not open webcam.")
        return

    print(f"[capture] category={args.category}  name={args.name}")
    print("[capture] SPACE = save frame, Q = quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            face_crop, bbox = detector.detect_and_crop(frame)
            display = frame.copy()
            if bbox is not None:
                detector.draw_bbox(display, bbox)
            else:
                cv2.putText(display, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("DeepCheck - Calibration Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if face_crop is None:
                    print("[capture] No face detected - not saved.")
                    continue
                idx = next_index(out_dir, args.name, args.category)
                out_path = os.path.join(out_dir, f"{args.name}_{args.category}_{idx}.jpg")
                cv2.imwrite(out_path, face_crop)
                print(f"[capture] saved {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
