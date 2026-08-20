"""
Analyzes calibration_data/<category>/ photos and suggests threshold values
for utils/constants.py, using the exact same feature extractors the app
uses at runtime (core/spatial_features.py, core/frequency_features.py).

Currently covers real vs. replay only - print is skipped for now (no
print photos collected yet). Anything that specifically needs a
real-vs-print boundary (OTSU_TEXTURE_THRESHOLD, ORB_MIN_KEYPOINTS,
REPLAY_FFT_RATIO_MIN as a print/replay split) is reported as "not
calibrated yet" rather than guessed from real-vs-replay data alone.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\analyze_calibration_data.py
"""
import glob
import os
import statistics
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spatial_features import SpatialFeatureExtractor
from core.frequency_features import FrequencyAnalyzer

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_data")

# Only these are analyzed right now - "print" is skipped until we collect
# printed-photo samples.
CATEGORIES = ("real", "replay")

FEATURE_NAMES = ("texture_var", "orb_keypoints", "fft_ratio", "edge_density")


def load_images(category):
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, category, "*.jpg")))
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[analyze] WARNING: could not read {path}, skipping.")
            continue
        images.append((path, img))
    return images


def extract_features(face_crop, spatial, frequency):
    _, edge_density = spatial.canny_edges(face_crop)
    texture_var, _ = spatial.otsu_texture_score(face_crop)
    orb_count, _ = spatial.orb_keypoint_score(face_crop)
    fft_ratio, _, _ = frequency.analyze(face_crop)
    return {
        "texture_var": texture_var,
        "orb_keypoints": orb_count,
        "fft_ratio": fft_ratio,
        "edge_density": edge_density,
    }


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def print_summary_table(per_category_features):
    print("\n=== Feature distributions ===")
    for feature in FEATURE_NAMES:
        print(f"\n--- {feature} ---")
        for category in CATEGORIES:
            values = per_category_features[category][feature]
            stats = summarize(values)
            if stats is None:
                print(f"  {category:8s}: no images found")
                continue
            print(
                f"  {category:8s}: n={stats['n']:2d}  "
                f"mean={stats['mean']:.4f}  stdev={stats['stdev']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}"
            )


def suggest_midpoint_threshold(real_values, replay_values, higher_means_real):
    """
    Suggest a threshold as the midpoint between the two classes' nearest
    edges. Returns (suggestion, separated) where `separated` is False when
    the two classes' ranges overlap (suggestion is still returned, but
    flagged as unreliable).
    """
    if not real_values or not replay_values:
        return None, False

    if higher_means_real:
        real_edge = min(real_values)
        replay_edge = max(replay_values)
        suggestion = (real_edge + replay_edge) / 2.0
        separated = real_edge > replay_edge
    else:
        real_edge = max(real_values)
        replay_edge = min(replay_values)
        suggestion = (real_edge + replay_edge) / 2.0
        separated = replay_edge > real_edge

    return suggestion, separated


def print_threshold_suggestions(per_category_features):
    print("\n=== Suggested thresholds (real vs. replay only) ===")

    real = per_category_features["real"]
    replay = per_category_features["replay"]

    # fft_ratio: higher = more likely spoof (replay), so "higher means real"
    # is False for the base FFT_HIGH_FREQ_RATIO split.
    fft_suggestion, fft_separated = suggest_midpoint_threshold(
        real["fft_ratio"], replay["fft_ratio"], higher_means_real=False
    )
    if fft_suggestion is not None:
        flag = "" if fft_separated else "  (WARNING: real/replay ranges overlap - not a clean split)"
        print(f"FFT_HIGH_FREQ_RATIO         ~ {fft_suggestion:.4f}{flag}")
    else:
        print("FFT_HIGH_FREQ_RATIO         not enough data")

    # texture_var / orb_keypoints / edge_density: higher = more likely real.
    texture_suggestion, texture_separated = suggest_midpoint_threshold(
        real["texture_var"], replay["texture_var"], higher_means_real=True
    )
    if texture_suggestion is not None:
        flag = "" if texture_separated else "  (WARNING: real/replay ranges overlap)"
        print(f"OTSU_TEXTURE_THRESHOLD      ~ {texture_suggestion:.2f} vs. real{flag}"
              "  [NOTE: this threshold is meant to catch PRINTS, not replays -"
              " treat this as informational only until print data exists]")
    else:
        print("OTSU_TEXTURE_THRESHOLD      not calibrated (no print data yet)")

    orb_suggestion, orb_separated = suggest_midpoint_threshold(
        real["orb_keypoints"], replay["orb_keypoints"], higher_means_real=True
    )
    if orb_suggestion is not None:
        flag = "" if orb_separated else "  (WARNING: real/replay ranges overlap)"
        print(f"ORB_MIN_KEYPOINTS           ~ {orb_suggestion:.1f} vs. real{flag}"
              "  [NOTE: same caveat - primarily a print detector, informational only]")
    else:
        print("ORB_MIN_KEYPOINTS           not calibrated (no print data yet)")

    print(
        "REPLAY_FFT_RATIO_MIN        not calibrated (needs print samples too, "
        "to separate print-vs-replay - skipped for now)"
    )

    # Rule-based real_confidence, using the SAME weights as
    # core/passive_classifier.py, so PASSIVE_ACCEPT/REJECT suggestions
    # match what the app actually computes.
    weights = {"texture": 0.35, "orb": 0.25, "fft": 0.25, "edge": 0.15}

    def real_conf(feat):
        texture_score = min(feat["texture_var"] / 500.0, 1.0)
        orb_score = min(feat["orb_keypoints"] / 200.0, 1.0)
        fft_score = 1.0 - min(feat["fft_ratio"] / 0.85, 1.0)
        edge_score = min(feat["edge_density"] / 0.15, 1.0)
        return (
            weights["texture"] * texture_score
            + weights["orb"] * orb_score
            + weights["fft"] * fft_score
            + weights["edge"] * edge_score
        )

    real_confs = [real_conf(f) for f in real["per_image"]]
    replay_confs = [real_conf(f) for f in replay["per_image"]]

    if real_confs and replay_confs:
        real_stats = summarize(real_confs)
        replay_stats = summarize(replay_confs)
        print(f"\nrule-based real_confidence: real mean={real_stats['mean']:.4f} "
              f"min={real_stats['min']:.4f}  |  replay mean={replay_stats['mean']:.4f} "
              f"max={replay_stats['max']:.4f}")
        accept_suggestion = max(replay_stats["max"], real_stats["mean"] * 0.9)
        reject_suggestion = min(real_stats["min"], replay_stats["mean"] * 1.1)
        if reject_suggestion > accept_suggestion:
            reject_suggestion, accept_suggestion = accept_suggestion, reject_suggestion
        print(f"PASSIVE_ACCEPT_THRESHOLD    ~ {accept_suggestion:.4f}")
        print(f"PASSIVE_REJECT_THRESHOLD    ~ {reject_suggestion:.4f}")
    else:
        print("\nPASSIVE_ACCEPT/REJECT_THRESHOLD  not enough data")


def main():
    spatial = SpatialFeatureExtractor()
    frequency = FrequencyAnalyzer()

    per_category_features = {
        category: {name: [] for name in FEATURE_NAMES} for category in CATEGORIES
    }
    for category in CATEGORIES:
        per_category_features[category]["per_image"] = []

    for category in CATEGORIES:
        images = load_images(category)
        if not images:
            print(f"[analyze] No images found in calibration_data/{category}/ - skipping.")
            continue
        print(f"[analyze] {category}: {len(images)} image(s)")
        for path, img in images:
            feats = extract_features(img, spatial, frequency)
            for name in FEATURE_NAMES:
                per_category_features[category][name].append(feats[name])
            per_category_features[category]["per_image"].append(feats)

    if not per_category_features["real"]["per_image"] or not per_category_features["replay"]["per_image"]:
        print("\n[analyze] Need at least one image in BOTH calibration_data/real/ "
              "and calibration_data/replay/ to compute suggestions.")
        return

    print_summary_table(per_category_features)
    print_threshold_suggestions(per_category_features)

    print(
        "\n[analyze] NOTE: 'print' category was skipped entirely (no samples "
        "collected yet). OTSU_TEXTURE_THRESHOLD, ORB_MIN_KEYPOINTS, and "
        "REPLAY_FFT_RATIO_MIN are print-oriented thresholds and should be "
        "re-run once print photos are added to calibration_data/print/."
    )


if __name__ == "__main__":
    main()
