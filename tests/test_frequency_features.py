import numpy as np

from core.frequency_features import FrequencyAnalyzer


def test_analyze_returns_ratio_in_valid_range():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    ratio, mag_display, magnitude = FrequencyAnalyzer().analyze(image)

    assert 0.0 <= ratio <= 1.0
    assert mag_display.shape == (224, 224, 3)
    assert magnitude.shape == (224, 224)


def test_analyze_flat_image_has_lower_high_freq_ratio_than_checkerboard():
    flat = np.full((224, 224, 3), 128, dtype=np.uint8)

    # Checkerboard pattern concentrates energy in high spatial frequencies,
    # mimicking the pixel-grid artifacts a screen replay produces.
    checker = np.zeros((224, 224, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255

    flat_ratio, _, _ = FrequencyAnalyzer().analyze(flat)
    checker_ratio, _, _ = FrequencyAnalyzer().analyze(checker)

    assert checker_ratio > flat_ratio
