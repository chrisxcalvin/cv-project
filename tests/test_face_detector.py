import cv2
import numpy as np

from core.face_detector import FaceDetector


def test_count_faces_returns_zero_for_none_frame():
    detector = FaceDetector()
    try:
        assert detector.count_faces(None) == 0
    finally:
        detector.close()


def test_count_faces_returns_zero_for_blank_frame():
    detector = FaceDetector()
    try:
        blank = np.full((224, 224, 3), 128, dtype=np.uint8)
        assert detector.count_faces(blank) == 0
    finally:
        detector.close()


def test_detect_and_crop_returns_none_for_blank_frame():
    detector = FaceDetector()
    try:
        blank = np.full((224, 224, 3), 128, dtype=np.uint8)
        crop, bbox = detector.detect_and_crop(blank)
        assert crop is None
        assert bbox is None
    finally:
        detector.close()


def test_detect_device_bezel_returns_false_for_none_inputs():
    detector = FaceDetector()
    try:
        frame = np.full((300, 300, 3), 128, dtype=np.uint8)
        detected, count = detector.detect_device_bezel(None, (10, 10, 50, 50, 0.9))
        assert detected is False and count == 0
        detected, count = detector.detect_device_bezel(frame, None)
        assert detected is False and count == 0
    finally:
        detector.close()


def test_detect_device_bezel_returns_false_for_flat_background():
    detector = FaceDetector()
    try:
        # A flat, featureless frame around the "face" box has no edges at
        # all, let alone long straight ones.
        frame = np.full((300, 300, 3), 128, dtype=np.uint8)
        bbox = (100, 100, 60, 60, 0.9)  # centered face box
        detected, count = detector.detect_device_bezel(frame, bbox)
        assert detected is False
        assert count == 0
    finally:
        detector.close()


def test_detect_device_bezel_detects_a_rectangular_border():
    detector = FaceDetector()
    try:
        # A bright rectangular border (phone/tablet bezel stand-in) drawn
        # around a centered "face" box, on a flat background otherwise -
        # long, straight, axis-aligned edges close to the face.
        frame = np.full((300, 300, 3), 30, dtype=np.uint8)
        # bbox below spans x/y [110,190]; margin_ratio=1.6 expands that to
        # an ROI of [86,214] in both axes, so this border (just outside the
        # face box, well inside that ROI) is what the check should see.
        cv2.rectangle(frame, (90, 90), (210, 210), (255, 255, 255), thickness=4)
        bbox = (110, 110, 80, 80, 0.9)  # centered inside the border
        detected, count = detector.detect_device_bezel(frame, bbox, margin_ratio=1.6)
        assert detected is True
        assert count >= 2
    finally:
        detector.close()
