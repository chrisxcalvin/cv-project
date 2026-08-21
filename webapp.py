# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py, webapp.py
"""
DeepCheck web demo - Flask backend.

Replaces the Streamlit dashboard (ui/dashboard.py) with a custom-designed
single-page web UI (templates/index.html + static/). Keeps the same
snapshot-based flow as the Streamlit version (browser camera -> single
photo -> run pipeline -> optional active-challenge photo -> verdict),
because that flow is what actually works reliably with a browser webcam -
only the presentation layer changes, core/ is untouched.

Run from the project root:
    .venv\\Scripts\\python.exe webapp.py

Then open http://localhost:5000 in a browser.
"""
import base64
import os
import secrets

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, session

from core.face_detector import FaceDetector
from core.passive_classifier import PassiveClassifier
from core.active_challenge import ActiveChallenge
from core.fusion import DecisionFusion
from utils import logger as db_logger
from utils.constants import (
    EAR_BLINK_THRESHOLD,
    HEAD_TURN_THRESHOLD,
    MOUTH_OPEN_THRESHOLD,
    MAX_FACES_ALLOWED,
    TEXTURE_SCORE_DIVISOR,
    FFT_SCORE_DIVISOR,
    ORB_SCORE_DIVISOR,
    DEVICE_BEZEL_PENALTY,
)

app = Flask(__name__)
# Session only ever holds small scalars (a float confidence, a challenge
# name, a retry counter) - never raw frames/images - so a signed cookie
# session is fine here; this is a local demo app, not a public deployment.
app.secret_key = secrets.token_hex(16)

# Singletons - loading the MediaPipe/CNN models is expensive, do it once at
# process start rather than per-request (mirrors ui/dashboard.py's
# @st.cache_resource pattern).
face_detector = FaceDetector()
classifier = PassiveClassifier()
active_challenge = ActiveChallenge()
fusion = DecisionFusion()

CHALLENGE_PROMPTS = {
    "blink": {"label": "Blink", "instruction": "Click capture, then blink naturally - it captures a short burst so timing doesn't need to be exact.", "icon": "eye"},
    "turn_left": {"label": "Turn Left", "instruction": "Click capture, then turn your head slightly LEFT and hold it briefly.", "icon": "arrow-left"},
    "turn_right": {"label": "Turn Right", "instruction": "Click capture, then turn your head slightly RIGHT and hold it briefly.", "icon": "arrow-right"},
    "mouth_open": {"label": "Open Mouth", "instruction": "Click capture, then open your mouth and hold it briefly.", "icon": "mouth"},
}


def decode_data_url(data_url: str):
    """Decode a 'data:image/jpeg;base64,...' string into a BGR ndarray."""
    header, _, b64data = data_url.partition(",")
    raw = base64.b64decode(b64data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image(img, is_gray=False) -> str:
    """Encode a BGR (or single-channel) ndarray into a 'data:image/jpeg;base64,...' string."""
    if img is None:
        return None
    if is_gray and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def passive_result_to_json(passive_result: dict) -> dict:
    # Display-only 0-1 bar magnitudes for the Feature Analysis panel. Uses
    # the same divisors as core/passive_classifier.py's rule-based scoring
    # (utils/constants.py), but WITHOUT the "more real" inversion applied
    # to fft there - this is just "how large is this raw measurement",
    # shown for both CNN and rule-based modes since the four features are
    # always computed either way (see PassiveClassifier.predict()).
    sub_scores = {
        "texture": min(passive_result["texture_var"] / TEXTURE_SCORE_DIVISOR, 1.0),
        "orb": min(passive_result["orb_keypoints"] / ORB_SCORE_DIVISOR, 1.0),
        "fft": min(passive_result["fft_ratio"] / FFT_SCORE_DIVISOR, 1.0),
        "edge": min(passive_result["edge_density"] / 0.15, 1.0),
    }

    return {
        "real_confidence": passive_result["real_confidence"],
        "texture_var": passive_result["texture_var"],
        "orb_keypoints": passive_result["orb_keypoints"],
        "fft_ratio": passive_result["fft_ratio"],
        "edge_density": passive_result["edge_density"],
        "method": passive_result["method"],
        "cnn_confidence": passive_result["cnn_confidence"],
        "spoof_type": passive_result["spoof_type"],
        "weighted_breakdown": passive_result["weighted_breakdown"],
        "sub_scores": sub_scores,
        "images": {
            "edges": encode_image(passive_result["edges"], is_gray=True),
            "orb_keypoints_img": encode_image(passive_result["kp_frame"]),
            "fft_spectrum": encode_image(passive_result["fft_display"]),
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/verify", methods=["POST"])
def api_verify():
    body = request.get_json(force=True)
    frame = decode_data_url(body["image"])
    if frame is None:
        return jsonify({"error": "Could not decode image."}), 400

    face_crop, bbox = face_detector.detect_and_crop(frame)
    num_faces = face_detector.count_faces(frame)

    if face_crop is None:
        return jsonify({"error": "no_face", "message": "No face detected. Recapture with your face clearly visible."})

    if num_faces > MAX_FACES_ALLOWED:
        return jsonify({
            "error": "multiple_faces",
            "message": f"{num_faces} faces detected - only one person may attempt verification at a time.",
        })

    passive_result = classifier.predict(face_crop)

    # Device-bezel check: looks in the region AROUND the face (not the
    # tight crop the rest of passive_result is based on) for long straight
    # edges consistent with a phone/tablet/monitor bezel. Penalty, not a
    # hard veto - see DEVICE_BEZEL_PENALTY in utils/constants.py.
    bezel_detected, bezel_line_count = face_detector.detect_device_bezel(frame, bbox)
    if bezel_detected:
        passive_result["real_confidence"] = round(
            max(0.0, passive_result["real_confidence"] - DEVICE_BEZEL_PENALTY), 4
        )

    decision = fusion.decide(passive_result)

    response = {
        "verdict": decision["verdict"],
        "confidence": decision["confidence"],
        "reason": decision.get("reason"),
        "passive": passive_result_to_json(passive_result),
        "face_crop": encode_image(face_crop),
        "bezel_detected": bezel_detected,
    }

    if decision["verdict"] == "PENDING":
        # DEMO MODE: force the always-easiest-to-perform-on-cue gesture
        # (blink) instead of a random pick from CHALLENGES. This is a UI
        # toggle, not a change to core/active_challenge.py - the real
        # random selection (pick_random_challenge) is untouched and still
        # what actually runs day-to-day / in tests. Scoped purely to make
        # a live demo predictable, not to change how the system works.
        if body.get("demo_mode"):
            challenge = "blink"
        else:
            challenge = active_challenge.pick_random_challenge()
        session["pending_challenge"] = challenge
        session["pending_confidence"] = passive_result["real_confidence"]
        session["challenge_retry_count"] = 0
        response["challenge"] = {"type": challenge, **CHALLENGE_PROMPTS[challenge]}
    else:
        session.pop("pending_challenge", None)
        session.pop("pending_confidence", None)
        session.pop("challenge_retry_count", None)
        db_logger.log_attempt(passive_result, decision, None)

    return jsonify(response)


@app.route("/api/challenge", methods=["POST"])
def api_challenge():
    challenge = session.get("pending_challenge")
    pending_confidence = session.get("pending_confidence")
    if challenge is None or pending_confidence is None:
        return jsonify({"error": "no_pending_challenge",
                         "message": "No active challenge in progress - capture your photo again."}), 400

    body = request.get_json(force=True)
    # `images`: a short burst of frames captured over ~1s (see app.js
    # captureBurst) rather than one single photo. A one-shot capture
    # forces the user to hit an exact millisecond - eyes closed AT THE
    # INSTANT the shutter fires for "blink", which is unnatural and
    # rejects genuine live users who blinked a moment before/after
    # clicking. Evaluating every frame in the burst and taking the first
    # one that actually satisfies the gesture fixes that without loosening
    # the underlying EAR/yaw/MAR thresholds at all. Falls back to a single
    # `image` for backward compatibility.
    image_list = body.get("images") or ([body["image"]] if "image" in body else [])
    frames = [decode_data_url(img) for img in image_list]
    frames = [f for f in frames if f is not None]
    if not frames:
        return jsonify({"error": "Could not decode image."}), 400

    glasses_mode = bool(body.get("glasses_mode"))

    # Requires N_FRAME_CONFIRM CONSECUTIVE frames to satisfy the gesture,
    # not just one lucky frame (see ActiveChallenge.evaluate_burst's
    # docstring) - a single-frame check let a tilted photo fake a head
    # turn during demo testing.
    active_result, confirmed_frames = active_challenge.evaluate_burst(
        frames, challenge, glasses_mode=glasses_mode
    )
    winning_frame = confirmed_frames[-1] if confirmed_frames else frames[-1]

    challenge_crop, _ = face_detector.detect_and_crop(winning_frame)
    if challenge_crop is None:
        return jsonify({"error": "no_face", "message": "No face detected in challenge photo."})

    if active_result["reason"] == "no_face_detected" and session.get("challenge_retry_count", 0) < 1:
        session["challenge_retry_count"] = session.get("challenge_retry_count", 0) + 1
        return jsonify({
            "verdict": "PENDING",
            "retry": True,
            "message": "Couldn't track facial landmarks on that photo. Keep more of your face visible and recapture.",
            "challenge": {"type": challenge, **CHALLENGE_PROMPTS[challenge]},
        })

    # Anti-swap check: re-score EVERY frame in the confirmed streak, not
    # just the winning one - a lone well-angled/low-glare frame is much
    # easier for a spoof to get lucky on than a whole streak. Taking the
    # minimum means the weakest frame in the streak decides. Each frame
    # also gets the same device-bezel penalty as /api/verify (see
    # DEVICE_BEZEL_PENALTY in utils/constants.py).
    challenge_confidences = []
    for f in confirmed_frames:
        crop, cbbox = face_detector.detect_and_crop(f)
        if crop is None:
            continue
        conf = classifier.predict(crop)["real_confidence"]
        if face_detector.detect_device_bezel(f, cbbox)[0]:
            conf = max(0.0, conf - DEVICE_BEZEL_PENALTY)
        challenge_confidences.append(conf)

    if challenge_confidences:
        challenge_passive_result = {"real_confidence": min(challenge_confidences)}
    else:
        challenge_passive_result = classifier.predict(challenge_crop)

    pending_passive_result = {"real_confidence": pending_confidence}
    decision = fusion.decide(pending_passive_result, active_result, challenge_passive_result)

    db_logger.log_attempt(pending_passive_result, decision, active_result, challenge_passive_result)

    session.pop("pending_challenge", None)
    session.pop("pending_confidence", None)
    session.pop("challenge_retry_count", None)

    thresholds = {
        "blink": f"EAR must drop below {EAR_BLINK_THRESHOLD:.2f} (yours: {active_result.get('ear_min')})",
        "turn_left": f"signed yaw offset must be >= {HEAD_TURN_THRESHOLD:.2f} (yours: {active_result.get('yaw_signed')})",
        "turn_right": f"signed yaw offset must be <= -{HEAD_TURN_THRESHOLD:.2f} (yours: {active_result.get('yaw_signed')})",
        "mouth_open": f"MAR must exceed {MOUTH_OPEN_THRESHOLD:.2f} (yours: {active_result.get('mar')})",
    }

    return jsonify({
        "verdict": decision["verdict"],
        "confidence": decision["confidence"],
        "reason": decision.get("reason"),
        "active_result": active_result,
        "measurement_note": thresholds.get(challenge),
        "challenge_photo_confidence": challenge_passive_result["real_confidence"],
        "challenge_photo_reject_threshold": DecisionFusion.CHALLENGE_REJECT_THRESHOLD,
    })


@app.route("/api/analytics")
def api_analytics():
    stats = db_logger.get_stats()
    recent = db_logger.get_recent(10)
    return jsonify({"stats": stats, "recent": recent})


if __name__ == "__main__":
    db_logger.init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
