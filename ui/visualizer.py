# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
OpenCV drawing helpers used by the --live overlay window (main.py --live).
Each function mutates and returns `frame` in place.
"""
import cv2
import numpy as np

VERDICT_COLORS = {
    "ACCEPT": (0, 200, 0),
    "REJECT": (0, 0, 220),
    "PENDING": (0, 200, 255),
}


def draw_verdict_overlay(frame, verdict, confidence):
    """Large ACCEPT (green) / REJECT (red) / PENDING (amber) banner at the top."""
    h, w = frame.shape[:2]
    color = VERDICT_COLORS.get(verdict, (200, 200, 200))

    banner_h = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = f"{verdict}  ({confidence * 100:0.1f}% real)"
    cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_feature_panel(frame, passive_result):
    """Small panel (bottom-left) showing the raw feature scores."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 150
    x0, y0 = 10, h - panel_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"method:     {passive_result.get('method', '-')}",
        f"texture_var:{passive_result.get('texture_var', '-')}",
        f"orb_kp:     {passive_result.get('orb_keypoints', '-')}",
        f"fft_ratio:  {passive_result.get('fft_ratio', '-')}",
        f"edge_dens:  {passive_result.get('edge_density', '-')}",
        f"spoof_type: {passive_result.get('spoof_type') or '-'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 10, y0 + 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_ear_graph(frame, ear_history, w, h):
    """Rolling EAR line graph drawn top-right, occupying a (w x h) region."""
    fh, fw = frame.shape[:2]
    x0, y0 = fw - w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "EAR", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if len(ear_history) < 2:
        return frame

    values = list(ear_history)
    max_ear = max(0.4, max(values))
    min_ear = 0.0
    plot_top = y0 + 22
    plot_h = h - 30

    points = []
    n = len(values)
    for i, val in enumerate(values):
        px = x0 + int(i / max(1, n - 1) * (w - 16)) + 8
        norm = (val - min_ear) / (max_ear - min_ear + 1e-6)
        py = plot_top + plot_h - int(norm * plot_h)
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

    return frame


def draw_challenge_prompt(frame, challenge_type, remaining_sec):
    """Big centred text prompt for the active challenge (e.g. 'BLINK NOW')."""
    if challenge_type is None:
        return frame

    h, w = frame.shape[:2]
    label = {
        "blink": "BLINK NOW",
        "turn_left": "TURN LEFT",
        "turn_right": "TURN RIGHT",
        "mouth_open": "OPEN YOUR MOUTH",
    }.get(challenge_type, challenge_type.upper())

    text = f"{label}  ({remaining_sec:0.1f}s)"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    x = (w - text_w) // 2
    y = h // 2

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3, cv2.LINE_AA)
    return frame


def draw_fft_inset(frame, fft_display):
    """Small FFT magnitude spectrum thumbnail, drawn in the bottom-right corner."""
    if fft_display is None:
        return frame

    fh, fw = frame.shape[:2]
    inset_w, inset_h = 140, 140
    thumb = cv2.resize(fft_display, (inset_w, inset_h))

    x0, y0 = fw - inset_w - 10, fh - inset_h - 10
    frame[y0:y0 + inset_h, x0:x0 + inset_w] = thumb
    cv2.rectangle(frame, (x0, y0), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1)
    cv2.putText(frame, "FFT", (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return frame
