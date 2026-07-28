# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Face localisation and cropping.

SYLLABUS: Object Detection - MediaPipe Face Detection locates the face in
the raw webcam frame before any other processing (spatial filtering, edge
detection, thresholding, keypoint/frequency analysis) is applied.
"""
import cv2
import numpy as np
import mediapipe as mp

from utils.constants import (
    FACE_DETECTION_MIN_CONFIDENCE,
    FACE_CROP_PADDING_RATIO,
    FACE_CROP_SIZE,
)

mp_face_detection = mp.solutions.face_detection


class FaceDetector:
    def __init__(self, min_confidence: float = FACE_DETECTION_MIN_CONFIDENCE):
        # SYLLABUS: Object detection applied - MediaPipe Face Detection for
        # fast, accurate face localisation on live webcam video.
        self._detector = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_confidence
        )

    def detect_and_crop(self, frame):
        """
        Detect the most confident face in `frame` and return a padded,
        resized crop plus its bounding box.

        Returns:
            (face_crop_bgr, bbox) where bbox = (x, y, w, h, confidence)
            or (None, None) if no face was found.
        """
        if frame is None or frame.size == 0:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return None, None

        # Pick the detection with the highest confidence score.
        best = max(results.detections, key=lambda d: d.score[0] if d.score else 0.0)
        rel_box = best.location_data.relative_bounding_box
        confidence = best.score[0] if best.score else 0.0

        x = rel_box.xmin * w
        y = rel_box.ymin * h
        bw = rel_box.width * w
        bh = rel_box.height * h

        # Add padding around the raw box.
        pad_w = bw * FACE_CROP_PADDING_RATIO
        pad_h = bh * FACE_CROP_PADDING_RATIO

        x1 = int(max(0, x - pad_w))
        y1 = int(max(0, y - pad_h))
        x2 = int(min(w, x + bw + pad_w))
        y2 = int(min(h, y + bh + pad_h))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        crop = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
        bbox = (x1, y1, x2 - x1, y2 - y1, float(confidence))
        return crop, bbox

    def count_faces(self, frame):
        """
        Return the number of faces detected in `frame`.

        Used to reject an attempt outright when more than one face is
        visible - e.g. someone holding a printed photo up next to their own
        face, which the single-highest-confidence crop in detect_and_crop()
        would otherwise silently ignore.
        """
        if frame is None or frame.size == 0:
            return 0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        return len(results.detections) if results.detections else 0

    def draw_bbox(self, frame, bbox):
        """Draw the detected bounding box + confidence score on `frame` in place."""
        if bbox is None:
            return frame
        x, y, w, h, confidence = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
        label = f"face {confidence:.2f}"
        cv2.putText(
            frame, label, (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
        )
        return frame

    def close(self):
        self._detector.close()
