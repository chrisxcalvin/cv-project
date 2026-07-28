# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Frequency-domain analysis of a face crop via 2D FFT.
"""
import cv2
import numpy as np


class FrequencyAnalyzer:
    def analyze(self, face_crop):
        """
        Compute the 2D FFT magnitude spectrum of `face_crop` and the ratio
        of energy concentrated in the high-frequency band.

        Returns:
            (high_freq_ratio, mag_display_bgr, raw_magnitude)
        """
        # SYLLABUS: Fourier Transform and Frequency Domain
        # Printed photos show halftone dot patterns; screen replays show a
        # pixel/subpixel grid. Both create characteristic HIGH-FREQUENCY
        # energy spikes in the 2D FFT that a genuine live face does not.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        f_transform = np.fft.fft2(gray.astype(float))
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shifted) + 1)

        # Compute ratio of energy in HIGH frequency bands vs total.
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        inner_radius = min(h, w) // 8  # low-freq centre circle

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_w) ** 2 + (Y - center_h) ** 2)

        high_freq_mask = dist > inner_radius

        total_energy = np.sum(magnitude)
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        high_freq_ratio = float(high_freq_energy / (total_energy + 1e-6))

        # Normalise magnitude for display (0-255).
        mag_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_display = cv2.cvtColor(mag_display, cv2.COLOR_GRAY2BGR)

        return high_freq_ratio, mag_display, magnitude
