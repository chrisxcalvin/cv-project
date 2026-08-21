"""
Analyzes calibration_data/<category>/ photos and suggests threshold values
for utils/constants.py.

Scores every image through PassiveClassifier.predict() directly (the exact
function webapp.py and ui/dashboard.py call at runtime) rather than
reimplementing the fusion formula here - a prior version of this script
duplicated the rule-based weights/divisors by hand and drifted out of sync
with utils/constants.py's recalibrated values (0.35/0.25/0.25/0.15 and
500.0/0.85 here vs. the real 0.55/0.10/0.25/0.10 and 4000.0/0.935 in
constants.py), silently producing suggestions fit to the WRONG formula.
Calling predict() directly makes that class of drift impossible, and
means these numbers automatically reflect the CNN+rule ensemble
(core/passive_classifier.py) whenever models/passive_spoof_model.pt is
present, not just the rule-based fallback.

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

from core.passive_classifier import PassiveClassifier

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


def print_threshold_suggestions(per_category_features, method_used):
    print(f"\n=== Suggested thresholds (real vs. replay only, method={method_used}) ===")

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

    real_confs = real["real_confidence"]
    replay_confs = replay["real_confidence"]

    if real_confs and replay_confs:
        real_stats = summarize(real_confs)
        replay_stats = summarize(replay_confs)
        print(f"\nreal_confidence ({method_used}): real mean={real_stats['mean']:.4f} "
              f"min={real_stats['min']:.4f}  |  replay mean={replay_stats['mean']:.4f} "
              f"max={replay_stats['max']:.4f}")

        accept_suggestion = max(replay_stats["max"], real_stats["mean"] * 0.9)
        reject_suggestion = min(real_stats["min"], replay_stats["mean"] * 1.1)
        if reject_suggestion > accept_suggestion:
            reject_suggestion, accept_suggestion = accept_suggestion, reject_suggestion
        print(f"PASSIVE_ACCEPT_THRESHOLD    ~ {accept_suggestion:.4f}")
        print(f"PASSIVE_REJECT_THRESHOLD    ~ {reject_suggestion:.4f}")

        # CHALLENGE_REJECT_THRESHOLD (utils/constants.py): the anti-swap
        # re-check run on the ACTIVE CHALLENGE photo (core/fusion.py), not
        # the user's first photo. It can afford to sit closer to the real
        # class than PASSIVE_REJECT_THRESHOLD does, since by that point
        # the user has already performed a real gesture - there's no
        # "don't punish an unconditioned first photo" excuse left, and its
        # whole job is catching a replay swapped in for the gesture.
        separated = real_stats["min"] > replay_stats["max"]
        if separated:
            gap = real_stats["min"] - replay_stats["max"]
            challenge_suggestion = replay_stats["max"] + gap * 0.5
            flag = ""
        else:
            # Overlapping ranges - no threshold cleanly separates the two
            # classes. Suggest just above the observed replay max as the
            # best available number, but say so plainly instead of
            # quietly proposing something that can't actually work.
            challenge_suggestion = replay_stats["max"] + 0.02
            flag = ("  (WARNING: real min <= replay max - ranges overlap, "
                    "NO threshold can cleanly separate these classes with "
                    "this feature set / data. This is the best available "
                    "number, not a safe guarantee.)")
        print(f"CHALLENGE_REJECT_THRESHOLD  ~ {challenge_suggestion:.4f}{flag}")
    else:
        print("\nPASSIVE_ACCEPT/REJECT_THRESHOLD  not enough data")
        print("CHALLENGE_REJECT_THRESHOLD       not enough data")


def main():
    classifier = PassiveClassifier()
    method_used = "rule_based"
    if classifier.use_cnn:
        print("[analyze] Note: a CNN checkpoint is loaded, but real_confidence "
              "is rule-based regardless (see PassiveClassifier.predict()'s "
              "CNN-mode branch for why) - cnn_confidence is computed but not "
              "part of these numbers.")

    per_category_features = {
        category: {name: [] for name in FEATURE_NAMES} for category in CATEGORIES
    }
    for category in CATEGORIES:
        per_category_features[category]["real_confidence"] = []
        per_category_features[category]["per_image"] = []

    for category in CATEGORIES:
        images = load_images(category)
        if not images:
            print(f"[analyze] No images found in calibration_data/{category}/ - skipping.")
            continue
        print(f"[analyze] {category}: {len(images)} image(s)")
        for path, img in images:
            result = classifier.predict(img)
            for name in FEATURE_NAMES:
                per_category_features[category][name].append(result[name])
            per_category_features[category]["real_confidence"].append(result["real_confidence"])
            per_category_features[category]["per_image"].append(result)

    if not per_category_features["real"]["per_image"] or not per_category_features["replay"]["per_image"]:
        print("\n[analyze] Need at least one image in BOTH calibration_data/real/ "
              "and calibration_data/replay/ to compute suggestions.")
        return

    print_summary_table(per_category_features)
    print_threshold_suggestions(per_category_features, method_used)

    print(
        "\n[analyze] NOTE: 'print' category was skipped entirely (no samples "
        "collected yet). OTSU_TEXTURE_THRESHOLD, ORB_MIN_KEYPOINTS, and "
        "REPLAY_FFT_RATIO_MIN are print-oriented thresholds and should be "
        "re-run once print photos are added to calibration_data/print/."
    )


if __name__ == "__main__":
    main()
