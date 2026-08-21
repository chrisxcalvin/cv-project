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
# Calibrated 2026-08-21 against calibration_data/ captured properly this
# time - BOTH real (8 images) and replay (7 images) through the SAME
# webcam pipeline (scripts/capture_calibration_photo.py), unlike an
# earlier same-day attempt that used professional headshots for "real"
# and got a meaningless comparison (see PASSIVE_FUSION_WEIGHTS below for
# the full story). With that dataset run through the recalibrated
# PASSIVE_FUSION_WEIGHTS/EDGE_SCORE_DIVISOR below, real_confidence
# measured: real = [0.4441, 0.6912] (one low outlier at 0.4441, the rest
# 0.555-0.691), replay = [0.1959, 0.6092] (one high outlier at 0.6092,
# the rest 0.196-0.300). Set comfortably BELOW the real distribution's
# worst case (including its outlier) so a genuine user's first photo is
# never hard-rejected, while still catching 6 of 7 replay samples
# outright without needing the active challenge. n=15 total, one phone -
# re-run scripts/analyze_calibration_data.py if the capture device
# changes materially.
PASSIVE_REJECT_THRESHOLD = 0.30

# Separate, stricter floor used ONLY when re-scoring the ACTIVE CHALLENGE
# photo (core/fusion.py's anti-swap check), not the user's first photo.
# PASSIVE_REJECT_THRESHOLD above is deliberately lenient because it must
# not punish a genuine live user's very first, unconditioned photo. The
# challenge photo is different: by the time it's captured, the user has
# already performed a real gesture, so there's no equivalent first-photo
# excuse for it to score low - and this is precisely the checkpoint meant
# to catch someone who showed a spoof first (landing in the uncertain
# band) and then held up a VIDEO of a real person performing the gesture
# to satisfy the active challenge (MediaPipe landmark tracking can't tell
# 2D screen playback from a live face - see core/active_challenge.py).
# Set a small margin above PASSIVE_REJECT_THRESHOLD (same calibration
# data, same reasoning - below the real distribution's worst case,
# catching most replay outright) rather than the identical value, to
# keep this check at least as strict as the first-photo one by
# construction. It also gets extra robustness the passive check above
# doesn't have: it's run against the MINIMUM score across a whole
# N_FRAME_CONFIRM streak of confirmed gesture frames (see
# ActiveChallenge.evaluate_burst), not a single photo, so a spoof would
# need to get lucky on 3 consecutive frames instead of 1 to slip through
# even the remaining overlap case (the single replay outlier, 0.6092).
CHALLENGE_REJECT_THRESHOLD = 0.35

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

# Rule-based fusion weights for PassiveClassifier.predict() - now the
# DECISION-DRIVING score even when a CNN checkpoint is loaded (see
# PassiveClassifier.predict()'s CNN-mode branch for why: the CNN scores
# replay HIGHER than real on average, corroborated across two independent
# calibration rounds, so blending it in only hurts - that finding stands
# independent of the weight-tuning below).
#
# History, because this constant flip-flopped twice in one day and the
# reasoning matters for whoever recalibrates it next:
#   1. Original: {texture:0.35, orb:0.25, fft:0.25, edge:0.15}, all voting
#      "higher = more real".
#   2. An early recalibration found orb/edge "backwards" for replay
#      (moire supposedly inflating them) and down-weighted both to 0.10.
#   3. A same-day re-recalibration measured the OPPOSITE using "real"
#      photos that turned out to be professional headshots, not webcam
#      captures (see calibration_data/README.md - this is exactly the
#      mismatch it warns against). Reverted.
#   4. THIS version: recalibrated against calibration_data/ captured
#      correctly this time (real AND replay both through
#      scripts/capture_calibration_photo.py, same webcam - real n=8,
#      replay n=7). texture_var and fft_ratio again showed weak/no
#      separation. orb_keypoints ALSO showed no reliable separation this
#      time (real 105-262, replay 131-297 - heavy overlap), disproving
#      round 3's finding once the comparison was apples-to-apples.
#      edge_density was the one feature that separated cleanly - but
#      INVERTED from the original assumption: HIGHER edge density meant
#      more likely REPLAY here (6 of 7 replay samples measured
#      0.084-0.117, all 8 real samples measured 0.026-0.052), matching
#      round 2's moire theory, not round 3's reversal of it. See
#      EDGE_SCORE_DIVISOR below and PassiveClassifier.predict() for the
#      inverted edge_score formula this weighting assumes.
#   Net: heavily weight the (inverted) edge signal, keep small non-zero
#   weights elsewhere as a hedge against overfitting an n=15 sample from
#   one phone/one day. Moire behavior is sensitive to the exact screen/
#   webcam/distance combination - re-run
#   scripts/analyze_calibration_data.py if the capture device changes.
PASSIVE_FUSION_WEIGHTS = {"texture": 0.25, "orb": 0.05, "fft": 0.10, "edge": 0.60}

# Normalization divisor for the (inverted) edge sub-score above -
# edge_score = 1.0 - min(edge_density / EDGE_SCORE_DIVISOR, 1.0), so
# higher edge_density -> lower edge_score -> more likely spoof. See
# PASSIVE_FUSION_WEIGHTS above for the full calibration writeup. 0.09
# sits between the observed real range (0.026-0.052) and the replay
# cluster (0.084-0.117, one outlier at 0.0385).
EDGE_SCORE_DIVISOR = 0.09

# Normalization divisor for the orb sub-score above (orb_score =
# min(orb_keypoints / ORB_SCORE_DIVISOR, 1.0)). Extracted from a bare
# 200.0 literal that used to live inline in passive_classifier.py. ORB's
# fusion weight is now small (see PASSIVE_FUSION_WEIGHTS above - it
# didn't separate real from replay reliably once measured correctly), so
# this divisor's exact value matters little; left at its original guess.
ORB_SCORE_DIVISOR = 200.0

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

# Added 2026-08-21 (demo-day fix): flat penalty subtracted from
# real_confidence when FaceDetector.detect_device_bezel() finds long,
# straight, axis-aligned lines close around the face - consistent with a
# phone/tablet/monitor bezel rather than a live face + background.
# Applied as a PENALTY, not a hard reject, on purpose: this heuristic has
# no calibration data behind it (no time to collect bezel/non-bezel
# samples before demo day), so a false positive (e.g. a picture frame or
# doorway edge near someone's head) should degrade a live user's score,
# not silently block them. 0.20 was chosen to meaningfully push a
# borderline replay attempt toward REJECT without single-handedly
# sinking a genuine real score (~0.55-0.69 per calibration_data/) below
# the uncertain band into an outright reject.
DEVICE_BEZEL_PENALTY = 0.20

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DB_PATH = "deepcheck_logs.db"
EAR_HISTORY_LEN = 60
