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
# Raised from 0.75. Real calibration data (calibration_data/, real vs.
# replay - see README there) showed the shipped CNN checkpoint
# (models/passive_spoof_model.pt, trained only on LFW + synthetic
# JPEG/halftone degradation, never on a real screen replay) scores replay
# photos HIGHER than real ones on average (replay mean 0.785 vs. real mean
# 0.737, replay max 0.925). At the old 0.75 floor, 19 of 32 replay photos
# (59%) in our sample would have auto-accepted, completely skipping the
# active blink/turn challenge - a real bypass, not just an accuracy gap.
# Real and replay confidence ranges overlap almost entirely (real
# 0.45-0.89, replay 0.65-0.93), so no threshold below ~0.93 is safe. Set
# above the observed replay max so passive alone can never auto-accept;
# the active challenge is the actual safety gate until the model is
# retrained on real presentation-attack photos instead of synthetic ones.
PASSIVE_ACCEPT_THRESHOLD = 0.95
# Lowered from 0.30 as a matching fix for the rule-based fallback path
# (used automatically if models/passive_spoof_model.pt is ever missing):
# with PASSIVE_FUSION_WEIGHTS/TEXTURE_SCORE_DIVISOR/FFT_SCORE_DIVISOR
# recalibrated to real webcam data (see below), real calibration photos
# legitimately scored as low as 0.18 under rule-based scoring - at 0.30,
# ~16% of real photos would have been hard-rejected with no chance at the
# active challenge. 0.18 sits just under the lowest real_confidence seen
# on real photos so far. Harmless to the current CNN path too, since real
# CNN scores never go anywhere near this low (min observed: 0.45).
PASSIVE_REJECT_THRESHOLD = 0.18

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

# Rule-based fusion weights for PassiveClassifier.predict() (used only when
# no CNN checkpoint is present). Originally 0.35/0.25/0.25/0.15
# (texture/orb/fft/edge), all voting "higher score = more real".
#
# Recalibrated after real/replay calibration data (see calibration_data/
# README.md) showed ORB keypoint count and Canny edge density are actively
# BACKWARDS for screen replay: filming a screen creates moire interference
# between the screen's sub-pixel grid and the webcam sensor, which *adds*
# dense fine-grained texture across the whole face rather than smoothing it
# out. A center-crop test (stripping the outer ~40% of each image, well
# past any phone bezel) made the gap bigger, not smaller, confirming the
# extra edges/keypoints come from the screen content itself, not capture
# artifacts. That assumption ("more edges/keypoints = more real") still
# plausibly holds for print vs. real (flat paper has no moire), so orb/edge
# are down-weighted rather than removed - see calibration_data/README.md
# for the full writeup and the print caveat (not yet calibrated, no print
# samples collected).
PASSIVE_FUSION_WEIGHTS = {"texture": 0.55, "orb": 0.10, "fft": 0.25, "edge": 0.10}

# Normalization divisors for the texture/fft sub-scores above
# (texture_score = min(texture_var / TEXTURE_SCORE_DIVISOR, 1.0), and
# similarly for fft). The original 500.0 / 0.85 values were guesses never
# checked against real webcam output: on actual calibration photos,
# texture_var runs ~900-4400 (so /500.0 saturated texture_score to 1.0 for
# EVERY image, real or replay) and fft_ratio runs ~0.919-0.933 (so /0.85
# saturated fft_score to 0.0 for every image). Both features were
# contributing constants, not signal - reweighting them alone (above) did
# nothing until these were fixed too. Recalibrated to the observed data
# range so the scores actually vary across images again.
TEXTURE_SCORE_DIVISOR = 4000.0
FFT_SCORE_DIVISOR = 0.935

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
