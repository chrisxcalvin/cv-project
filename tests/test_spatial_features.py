import numpy as np

from core.spatial_features import SpatialFeatureExtractor


def _flat_image():
    """A perfectly flat gray crop - the 'suspiciously flat, likely-printed' case."""
    return np.full((224, 224, 3), 128, dtype=np.uint8)


def _textured_image():
    """Random noise crop - stands in for a texture-rich, real-face-like crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def test_preprocess_returns_same_shape():
    extractor = SpatialFeatureExtractor()
    blurred, sharpened = extractor.preprocess(_textured_image())
    assert blurred.shape == (224, 224, 3)
    assert sharpened.shape == (224, 224, 3)


def test_canny_edges_flat_image_has_near_zero_density():
    extractor = SpatialFeatureExtractor()
    edges, density = extractor.canny_edges(_flat_image())
    assert edges.shape == (224, 224)
    assert density == 0.0


def test_canny_edges_textured_image_has_higher_density_than_flat():
    extractor = SpatialFeatureExtractor()
    _, flat_density = extractor.canny_edges(_flat_image())
    _, textured_density = extractor.canny_edges(_textured_image())
    assert textured_density > flat_density


def test_otsu_texture_score_flat_image_has_zero_variance():
    extractor = SpatialFeatureExtractor()
    variance, _ = extractor.otsu_texture_score(_flat_image())
    assert variance == 0.0


def test_otsu_texture_score_textured_image_has_higher_variance_than_flat():
    extractor = SpatialFeatureExtractor()
    flat_variance, _ = extractor.otsu_texture_score(_flat_image())
    textured_variance, _ = extractor.otsu_texture_score(_textured_image())
    assert textured_variance > flat_variance


def test_orb_keypoint_score_flat_image_has_few_or_no_keypoints():
    extractor = SpatialFeatureExtractor()
    count, kp_frame = extractor.orb_keypoint_score(_flat_image())
    assert count == 0
    assert kp_frame.shape == (224, 224, 3)


def test_orb_keypoint_score_textured_image_has_more_keypoints_than_flat():
    extractor = SpatialFeatureExtractor()
    flat_count, _ = extractor.orb_keypoint_score(_flat_image())
    textured_count, _ = extractor.orb_keypoint_score(_textured_image())
    assert textured_count > flat_count
