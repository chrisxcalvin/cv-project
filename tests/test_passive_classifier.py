import os

import numpy as np
import pytest

from core.passive_classifier import PassiveClassifier, classify_spoof_type
from utils.constants import PASSIVE_REJECT_THRESHOLD, OTSU_TEXTURE_THRESHOLD, REPLAY_FFT_RATIO_MIN

_CHECKPOINT = "models/passive_spoof_model.pt"


def _dummy_crop():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def test_falls_back_to_rule_based_when_no_model_file():
    clf = PassiveClassifier(model_path="models/does_not_exist.pt")
    assert clf.use_cnn is False

    result = clf.predict(_dummy_crop())
    assert result["method"] == "rule_based"
    assert 0.0 <= result["real_confidence"] <= 1.0


@pytest.mark.skipif(not os.path.exists(_CHECKPOINT),
                     reason="toy checkpoint not present - run scripts/train_toy_model.py first")
def test_uses_cnn_when_model_file_present():
    clf = PassiveClassifier(model_path="models/passive_spoof_model.pt")
    assert clf.use_cnn is True

    result = clf.predict(_dummy_crop())
    # real_confidence stays rule-based even with a checkpoint loaded - see
    # PassiveClassifier.predict()'s CNN-mode branch for why (calibration
    # data showed the CNN's errors aren't one-directional, so blending it
    # in hurts more than it helps). The CNN's own score is still exposed,
    # just not decision-driving.
    assert result["method"] == "rule_based"
    assert 0.0 <= result["real_confidence"] <= 1.0
    assert result["cnn_confidence"] is not None
    assert 0.0 <= result["cnn_confidence"] <= 1.0


def test_cnn_confidence_is_none_without_a_checkpoint():
    clf = PassiveClassifier(model_path="models/does_not_exist.pt")
    result = clf.predict(_dummy_crop())
    assert result["cnn_confidence"] is None


def test_predict_result_has_expected_keys_regardless_of_method():
    clf = PassiveClassifier(model_path="models/does_not_exist.pt")
    result = clf.predict(_dummy_crop())
    expected_keys = {
        "real_confidence", "texture_var", "orb_keypoints", "fft_ratio",
        "edge_density", "method", "spoof_type", "weighted_breakdown",
        "cnn_confidence", "kp_frame", "fft_display", "edges",
    }
    assert expected_keys.issubset(result.keys())


def test_classify_spoof_type_returns_none_when_confidence_not_spoof_leaning():
    assert classify_spoof_type(PASSIVE_REJECT_THRESHOLD + 0.01, texture_var=5.0, fft_ratio=0.5) is None


def test_classify_spoof_type_high_fft_ratio_reads_as_replay():
    result = classify_spoof_type(
        PASSIVE_REJECT_THRESHOLD - 0.01, texture_var=200.0, fft_ratio=REPLAY_FFT_RATIO_MIN + 0.01
    )
    assert result == "screen_replay"


def test_classify_spoof_type_flat_texture_low_fft_reads_as_print():
    result = classify_spoof_type(
        PASSIVE_REJECT_THRESHOLD - 0.01,
        texture_var=OTSU_TEXTURE_THRESHOLD - 1.0,
        fft_ratio=REPLAY_FFT_RATIO_MIN - 0.01,
    )
    assert result == "printed_photo"


def test_classify_spoof_type_ambiguous_case_reads_as_unclear():
    result = classify_spoof_type(
        PASSIVE_REJECT_THRESHOLD - 0.01,
        texture_var=OTSU_TEXTURE_THRESHOLD + 1.0,
        fft_ratio=REPLAY_FFT_RATIO_MIN - 0.01,
    )
    assert result == "unclear_spoof_type"
