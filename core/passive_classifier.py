# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Passive spoof classifier.

Works WITHOUT a trained model: when models/passive_spoof_model.pt is not
present, real_confidence is produced by a rule-based fusion of the Otsu
texture score, ORB keypoint count, FFT high-frequency ratio, and Canny edge
density (SYLLABUS: Decision Fusion at the feature level, distinct from the
higher-level passive+active fusion in core/fusion.py).

When the model file IS present, a MobileNetV2 classifier is used instead.
"""
import os
import cv2

from core.spatial_features import SpatialFeatureExtractor
from core.frequency_features import FrequencyAnalyzer
from utils.constants import (
    PASSIVE_REJECT_THRESHOLD,
    OTSU_TEXTURE_THRESHOLD,
    REPLAY_FFT_RATIO_MIN,
    PASSIVE_FUSION_WEIGHTS,
    TEXTURE_SCORE_DIVISOR,
    FFT_SCORE_DIVISOR,
)


def classify_spoof_type(real_confidence, texture_var, fft_ratio):
    """
    Sub-classify a likely-spoof result as a print photo or a screen replay.

    SYLLABUS: Decision Fusion (feature-level) - reuses the Otsu texture
    variance and FFT high-frequency ratio already computed for the primary
    real/spoof score, just with a second decision boundary applied only when
    the primary score already leans "spoof".

    Heuristic and illustrative only, like the rest of the rule-based fusion -
    see REPLAY_FFT_RATIO_MIN / OTSU_TEXTURE_THRESHOLD in utils/constants.py.
    Returns one of: "printed_photo", "screen_replay", "unclear_spoof_type",
    or None (when real_confidence doesn't indicate a spoof at all).
    """
    if real_confidence > PASSIVE_REJECT_THRESHOLD:
        return None

    if fft_ratio >= REPLAY_FFT_RATIO_MIN:
        return "screen_replay"
    if texture_var <= OTSU_TEXTURE_THRESHOLD:
        return "printed_photo"
    return "unclear_spoof_type"


class PassiveClassifier:
    def __init__(self, model_path="models/passive_spoof_model.pt"):
        self.spatial = SpatialFeatureExtractor()
        self.frequency = FrequencyAnalyzer()
        self.model = None
        self.use_cnn = False

        if os.path.exists(model_path):
            # Load MobileNetV2 for CNN-based classification.
            self.model = self._load_model(model_path)
            self.use_cnn = True
            print("[PassiveClassifier] CNN model loaded.")
        else:
            print("[PassiveClassifier] No model found - using rule-based fusion.")

    def predict(self, face_crop) -> dict:
        """
        Returns:
          {
            "real_confidence": float,   0.0 (spoof) to 1.0 (real)
            "texture_var":     float,
            "orb_keypoints":   int,
            "fft_ratio":       float,
            "edge_density":    float,
            "method":          "cnn" | "rule_based"
            "spoof_type":      "printed_photo" | "screen_replay" |
                               "unclear_spoof_type" | None (None if not spoof-leaning)
            "weighted_breakdown": dict[str, float] | None (None in "cnn" mode -
                               CNN scores have no per-feature breakdown to show)
            "kp_frame":        np.ndarray (BGR),
            "fft_display":     np.ndarray (BGR),
            "edges":           np.ndarray (single-channel edge map),
          }
        """
        # Always extract spatial and frequency features.
        blurred, sharpened = self.spatial.preprocess(face_crop)
        edges, edge_density = self.spatial.canny_edges(face_crop)
        texture_var, _ = self.spatial.otsu_texture_score(face_crop)
        orb_count, kp_frame = self.spatial.orb_keypoint_score(face_crop)
        fft_ratio, fft_display, _ = self.frequency.analyze(face_crop)

        weighted_breakdown = None
        if self.use_cnn:
            # NOTE: must use the raw face_crop, not `sharpened` - the toy
            # checkpoint (scripts/train_toy_model.py) was trained on plain
            # resized RGB crops, never on blurred+Laplacian-sharpened
            # images. Feeding `sharpened` here would silently move every
            # inference off the model's training distribution.
            real_conf = self._cnn_predict(face_crop)
            method = "cnn"
        else:
            # RULE-BASED FUSION (works without any training).
            # Each feature votes: higher score = more likely REAL.
            texture_score = min(texture_var / TEXTURE_SCORE_DIVISOR, 1.0)   # variance -> 0-1
            orb_score = min(orb_count / 200.0, 1.0)             # keypoints -> 0-1
            fft_score = 1.0 - min(fft_ratio / FFT_SCORE_DIVISOR, 1.0)  # inv: more high-freq = more spoof
            edge_score = min(edge_density / 0.15, 1.0)          # edge density -> 0-1

            # orb/edge are down-weighted vs. texture/fft - real calibration
            # data showed they're backwards for screen replay (moire
            # inflates keypoints/edges instead of reducing them). See
            # PASSIVE_FUSION_WEIGHTS in utils/constants.py for the writeup.
            weights = PASSIVE_FUSION_WEIGHTS
            sub_scores = {"texture": texture_score, "orb": orb_score,
                          "fft": fft_score, "edge": edge_score}

            # SYLLABUS: Decision Fusion - weighted combination of the Otsu
            # texture score, ORB keypoint count, FFT high-frequency energy,
            # and Canny edge density into one real/spoof confidence value.
            real_conf = sum(weights[k] * sub_scores[k] for k in weights)
            method = "rule_based"

            # Explainability breakdown: how many of the final real_conf
            # points each feature actually contributed, not just its raw
            # 0-1 sub-score - lets the dashboard show *why* the number came
            # out the way it did instead of just the number itself.
            weighted_breakdown = {
                "texture": round(weights["texture"] * sub_scores["texture"], 4),
                "orb": round(weights["orb"] * sub_scores["orb"], 4),
                "fft": round(weights["fft"] * sub_scores["fft"], 4),
                "edge": round(weights["edge"] * sub_scores["edge"], 4),
            }

        real_conf = round(float(real_conf), 4)
        spoof_type = classify_spoof_type(real_conf, texture_var, fft_ratio)

        return {
            "real_confidence": real_conf,
            "texture_var": round(float(texture_var), 2),
            "orb_keypoints": int(orb_count),
            "fft_ratio": round(float(fft_ratio), 4),
            "edge_density": round(float(edge_density), 4),
            "method": method,
            "spoof_type": spoof_type,
            "weighted_breakdown": weighted_breakdown,
            "kp_frame": kp_frame,
            "fft_display": fft_display,
            "edges": edges,
        }

    def _load_model(self, path):
        import torch
        import torch.nn as nn
        from torchvision import models

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
        model.eval()
        return model

    def _cnn_predict(self, face_crop):
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
        return probs[0].item()  # index 0 = real (adjust if class order differs)
