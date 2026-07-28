# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Central configuration file - all thresholds, landmark indices, and tunables
live here so every module reads from a single source of truth.
"""

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# (indices refer to the 468-point Face Mesh topology)
# ---------------------------------------------------------------------------

# SYLLABUS: EAR / Landmark tracking - six-point eye contours used in the
# standard Eye Aspect Ratio formula (Soukupova & Cech, 2016)
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]

# Full eye ring landmarks, used only for drawing/visualisation overlays
LEFT_EYE_RING_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                      466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_RING_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                       173, 157, 158, 159, 160, 161, 246]

# SYLLABUS: Head pose estimation - nose tip vs cheek-midpoint yaw offset
NOSE_TIP_IDX = 1
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454
CHIN_IDX = 152

# Mouth landmarks used for Mouth Aspect Ratio (MAR), same pattern as EAR:
# [left_corner, upper_inner_lip, right_corner, lower_inner_lip]
MOUTH_MAR_IDX = [78, 13, 308, 14]

# ---------------------------------------------------------------------------
# Active liveness challenge thresholds
# ---------------------------------------------------------------------------
EAR_BLINK_THRESHOLD = 0.21
EAR_BLINK_THRESHOLD_GLASSES = 0.24
HEAD_TURN_THRESHOLD = 0.07
# Heuristic, not empirically calibrated (same caveat as EAR_BLINK_THRESHOLD) -
# a closed/resting mouth typically measures MAR ~0.05-0.15, an open mouth/
# smile-with-teeth-shown typically exceeds ~0.5.
MOUTH_OPEN_THRESHOLD = 0.5
N_FRAME_CONFIRM = 3
CHALLENGE_TIMEOUT = 5.0

# ---------------------------------------------------------------------------
# Passive classifier / decision fusion thresholds
# ---------------------------------------------------------------------------
PASSIVE_ACCEPT_THRESHOLD = 0.75
PASSIVE_REJECT_THRESHOLD = 0.30

# SYLLABUS: Thresholding (Otsu) - texture-variance floor below which a face
# crop is considered a suspiciously flat region (likely a printed photo)
OTSU_TEXTURE_THRESHOLD = 15.0

# SYLLABUS: Feature Detection (ORB) - minimum strong keypoints expected on a
# genuine, texture-rich live face
ORB_MIN_KEYPOINTS = 80

# SYLLABUS: Frequency Domain (FFT) - high-frequency energy ratio above which
# moire / halftone / pixel-grid artifacts are suspected
FFT_HIGH_FREQ_RATIO = 0.15

# Spoof-type sub-classification (print vs. screen replay), used only once
# real_confidence has already flagged something as likely-spoof. Heuristic
# and illustrative, like the thresholds above - not fit to a labelled
# print-vs-replay attack dataset, since this project has never had access
# to one. A screen's RGB sub-pixel grid tends to concentrate MORE high-
# frequency FFT energy than a printed halftone dot pattern at typical
# webcam capture distance, so a ratio clearly above the base
# FFT_HIGH_FREQ_RATIO leans "replay"; a very flat texture below
# OTSU_TEXTURE_THRESHOLD with only moderately elevated FFT energy leans
# "print".
REPLAY_FFT_RATIO_MIN = FFT_HIGH_FREQ_RATIO * 1.6

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
FACE_DETECTION_MIN_CONFIDENCE = 0.6
FACE_CROP_PADDING_RATIO = 0.20
FACE_CROP_SIZE = 224
MAX_FACES_ALLOWED = 1  # reject frames showing more than one face (e.g. a
                        # photo held up beside the real presenter's face)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DB_PATH = "deepcheck_logs.db"
EAR_HISTORY_LEN = 60
