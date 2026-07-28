# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Spatial-domain feature extraction: smoothing, sharpening, edge detection,
Otsu thresholding, and ORB keypoint description on a face crop.
"""
import cv2
import numpy as np


class SpatialFeatureExtractor:
    def preprocess(self, face_crop):
        """Return (blurred, sharpened) versions of `face_crop`."""
        # SYLLABUS: Smoothing Spatial Filter
        # Gaussian blur (5x5 kernel, sigma=1.0) reduces webcam sensor noise
        # before any downstream feature extraction runs.
        blurred = cv2.GaussianBlur(face_crop, (5, 5), 1.0)

        # SYLLABUS: Sharpening Spatial Filter
        # Laplacian-style sharpening kernel enhances fine edge structure.
        # Real faces carry rich fine-grained edges (pores, hair, eyelashes);
        # printed photos / screen replays tend to look smoother after blur,
        # so sharpening exaggerates the difference.
        laplacian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(blurred, -1, laplacian_kernel)
        return blurred, sharpened

    def canny_edges(self, face_crop):
        """Return (edge_map, edge_density) for `face_crop`."""
        # SYLLABUS: Edge Detection
        # Canny edge detection on the face crop. Real faces show many
        # complex edges (pores, hair, eyelashes); printed photos / screens
        # show fewer, smoother edges.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        return edges, edge_density

    def otsu_texture_score(self, face_crop):
        """Return (texture_variance, otsu_thresholded_image) for `face_crop`."""
        # SYLLABUS: Thresholding - Otsu's method
        # Otsu automatically finds the optimal binary threshold; the raw
        # grayscale variance is used as a fast texture pre-screen - a very
        # low variance means large flat regions, a signal typical of a
        # printed photo rather than a real, texture-rich face.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        texture_variance = float(np.var(gray.astype(float)))
        return texture_variance, thresh

    def orb_keypoint_score(self, face_crop):
        """Return (keypoint_count, frame_with_keypoints_drawn) for `face_crop`."""
        # SYLLABUS: Feature Detection and Description - ORB
        # ORB (Oriented FAST and Rotated BRIEF) detects and describes local
        # keypoints. Real faces yield many rich, irregularly distributed
        # keypoints; printed photos and screens tend to yield fewer, or
        # regular grid-like patterns from the printing/pixel structure.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        kp_count = len(keypoints) if keypoints else 0
        kp_frame = cv2.drawKeypoints(
            face_crop.copy(), keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return kp_count, kp_frame
