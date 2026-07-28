```
SYLLABUS TECHNIQUES USED IN THIS PROJECT
==========================================
1. Smoothing Spatial Filter     -> core/spatial_features.py :: preprocess() - Gaussian blur
2. Sharpening Spatial Filter    -> core/spatial_features.py :: preprocess() - Laplacian
3. Edge Detection (Canny)       -> core/spatial_features.py :: canny_edges()
4. Thresholding (Otsu)          -> core/spatial_features.py :: otsu_texture_score()
5. Feature Detection (ORB)      -> core/spatial_features.py :: orb_keypoint_score()
6. Fourier Transform / FFT      -> core/frequency_features.py :: analyze()
7. Object Detection (face)      -> core/face_detector.py - MediaPipe Face Detection
8. EAR / Landmark tracking      -> core/active_challenge.py - MediaPipe Face Mesh (EAR blink + MAR mouth-open)
9. Decision Fusion              -> core/fusion.py - weighted multi-feature fusion
```

# DeepCheck

A webcam-based, real-time face liveness and anti-spoofing detection system built for a
Computer Vision course demo. DeepCheck combines a **passive** spoof classifier (texture,
edge, keypoint, and frequency analysis) with an **active** liveness challenge (blink /
head-turn) to decide whether a face in front of the camera is real or a print/screen replay.

## Setup

```
pip install -r requirements.txt
python main.py
```

That's it - no other setup steps needed on a clean Python install. If your machine
already has a different `mediapipe`/`tensorflow`/`opencv` combo installed globally
(common on shared ML dev machines), use a virtual environment first to avoid version
conflicts:

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python main.py
```

Note: `requirements.txt` pins `mediapipe==0.10.21` deliberately - MediaPipe 1.0.0
removed the legacy `mp.solutions` API (Face Detection / Face Mesh) that this project
relies on in favor of a newer Tasks API requiring separately downloaded model bundles,
so the pin keeps the project fully offline-runnable.

`python main.py` launches the Streamlit dashboard by
default. To run the OpenCV live webcam window instead:

```
python main.py --live
```

The system works out of the box **without any trained model file**: if
`models/passive_spoof_model.pt` is absent, `core/passive_classifier.py` automatically
falls back to a rule-based fusion of Otsu texture variance, ORB keypoint count, FFT
high-frequency ratio, and Canny edge density. If a trained MobileNetV2 checkpoint is
later placed at that path, the classifier switches to CNN inference automatically.

### Toy CNN checkpoint (`models/passive_spoof_model.pt`)

A trained checkpoint IS included, produced by `scripts/train_toy_model.py`. Be
straight about what it is when presenting it:

- **"Real" class** = LFW - Labeled Faces in the Wild (public research dataset, real
  color photographs, downloaded via `sklearn`, no signup/license gate).
- **"Spoof" class** = the SAME photos, synthetically degraded to mimic what a print
  or screen replay does to an image (JPEG recompression, halftone dot overlay,
  downsample/upsample blockiness). These are NOT real presentation-attack photos.
- This proves the CNN code path (checkpoint -> load -> inference) works end-to-end
  on realistic, full-color webcam-like input. It has **not** been validated against
  real print/replay attacks, and the ~92% validation accuracy is only meaningful on
  this synthetic task - not a real-world accuracy claim.
- Retrain it yourself with `.venv\Scripts\python.exe scripts\train_toy_model.py`
  (takes a few minutes; downloads ~200MB of LFW data on first run).

## How it works

**Layer 1 - Passive spoof detection.** Every captured face crop is scored on four
independent cues that separate real faces from printed photos / screen replays:
texture flatness (Otsu), keypoint richness (ORB), high-frequency artifacts (FFT), and
edge complexity (Canny). These are fused into a single `real_confidence` score, plus
(when rule-based) a per-feature **weighted-contribution breakdown** used for the
dashboard's explainability chart. When `real_confidence` already leans spoof,
`classify_spoof_type()` further sub-classifies it as a likely **printed photo** or
**screen replay** by re-checking the FFT ratio and texture variance against a second,
tighter boundary (heuristic, same caveat as the rest of the rule-based fusion).

**Layer 2 - Active liveness challenge.** If `real_confidence` lands in the uncertain
band (0.30 - 0.75), the system prompts the user to blink, turn left, turn right, or
open their mouth. MediaPipe Face Mesh landmarks drive Eye Aspect Ratio (EAR) blink
detection, a nose-tip-vs-cheek yaw estimate for head-turn detection, and Mouth Aspect
Ratio (MAR) for the mouth-open challenge - each confirmed over multiple consecutive
frames to reject single-frame noise. A frame showing more than one face is rejected
outright (blocks holding a spoof photo up next to your own real face).

**Layer 3 - Dashboard.** A Streamlit UI shows the live capture, feature visualizations
(edges / keypoints / FFT spectrum), scores, an explainability bar chart, a
session-long confidence-trend line graph, challenge prompts, the final ACCEPT/REJECT
verdict, a downloadable one-page PNG verdict report, and an Analytics tab reading from
a SQLite attempt log.

## Syllabus technique mapping

| # | Technique | Where it's implemented |
|---|-----------|------------------------|
| 1 | Smoothing spatial filter (Gaussian blur) | `core/spatial_features.py :: preprocess()` |
| 2 | Sharpening spatial filter (Laplacian) | `core/spatial_features.py :: preprocess()` |
| 3 | Edge detection (Canny) | `core/spatial_features.py :: canny_edges()` |
| 4 | Thresholding (Otsu) | `core/spatial_features.py :: otsu_texture_score()` |
| 5 | Feature detection & description (ORB) | `core/spatial_features.py :: orb_keypoint_score()` |
| 6 | Frequency domain (2D FFT) | `core/frequency_features.py :: analyze()` |
| 7 | Object/face detection | `core/face_detector.py` (MediaPipe Face Detection) |
| 8 | EAR + head pose + MAR (landmarks) | `core/active_challenge.py` (MediaPipe Face Mesh) |
| 9 | Decision fusion | `core/passive_classifier.py`, `core/fusion.py` |

## File-by-file description

```
deepcheck/
  main.py                    Entry point. Default: launches Streamlit. --live: OpenCV window.
  requirements.txt           Python dependencies.

  core/
    face_detector.py         MediaPipe face detection + padded crop + bbox drawing +
                              count_faces() multi-face guard.
    spatial_features.py      Gaussian blur, Laplacian sharpen, Canny, Otsu, ORB.
    frequency_features.py    2D FFT magnitude spectrum + high-frequency energy ratio.
    passive_classifier.py    Rule-based fusion (default) or MobileNetV2 CNN (if trained) +
                              classify_spoof_type() print-vs-replay sub-classification +
                              weighted-contribution breakdown for explainability.
    active_challenge.py      EAR blink + head-turn + MAR mouth-open detection,
                              N-frame confirmation.
    fusion.py                Final ACCEPT/REJECT/PENDING decision from passive + active.

  ui/
    dashboard.py             Streamlit app: Live Demo tab (feature views, explainability
                              chart, session confidence trend, downloadable verdict
                              report PNG) + Analytics tab.
    visualizer.py             OpenCV overlay drawing helpers for --live mode.

  utils/
    constants.py              All thresholds, landmark indices, config in one place.
    logger.py                 SQLite attempt logging + analytics queries.

  scripts/
    train_toy_model.py        Trains the toy MobileNetV2 checkpoint (see above).

  tests/                     pytest suite - fusion thresholds, spatial/frequency feature
                              math, passive classifier CNN/rule-based switch + spoof-type
                              classification, EAR/MAR math, face-detector edge cases.

  models/                    Trained .pt checkpoint (see "Toy CNN checkpoint" above).
  demo_scenarios/
    DEMO_SCRIPT.md            Demo-day walkthrough script.
```

## Testing

```
pip install -r requirements.txt   # includes pytest
pytest tests -v
```

Runs entirely offline against synthetic images/landmarks - no webcam or trained model
required (the one test that needs the CNN checkpoint skips gracefully if
`models/passive_spoof_model.pt` is missing).

## Demo day script

See [`demo_scenarios/DEMO_SCRIPT.md`](demo_scenarios/DEMO_SCRIPT.md) for the full
4-scenario walkthrough (real face / print attack / replay attack / borderline case).

## Team split

```
Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
  - Passive feature extraction (Otsu, ORB, FFT, Canny) and their rule-based fusion.
  - classify_spoof_type(): print-vs-replay sub-classification once a result leans spoof.
  - scripts/train_toy_model.py: toy MobileNetV2 checkpoint (LFW real class + synthetic
    print/replay degradation), wired into the existing CNN-vs-rule-based switch.

Person B: core/active_challenge.py, core/face_detector.py
  - EAR blink and yaw-based head-turn detection, N-frame confirmation.
  - Mouth Aspect Ratio (MAR) mouth-open challenge (3rd active liveness check).
  - Multi-face guard (count_faces()): rejects frames showing more than one face.

Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
  - Passive + active decision fusion (ACCEPT/REJECT/PENDING).
  - Dashboard: explainability weighted-contribution chart, session confidence-trend
    graph, downloadable one-page PNG verdict report.
  - SQLite logging extended to persist spoof_type per attempt.

tests/ (all three): each person's area has corresponding pytest coverage.
```
