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
