# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
Top-level decision fusion: combines the passive classifier's confidence with
an optional active-challenge result into one final verdict.
"""
from utils.constants import (
    PASSIVE_ACCEPT_THRESHOLD,
    PASSIVE_REJECT_THRESHOLD,
    CHALLENGE_REJECT_THRESHOLD,
)


class DecisionFusion:
    ACCEPT_THRESHOLD = PASSIVE_ACCEPT_THRESHOLD
    REJECT_THRESHOLD = PASSIVE_REJECT_THRESHOLD
    CHALLENGE_REJECT_THRESHOLD = CHALLENGE_REJECT_THRESHOLD

    def decide(self, passive_result: dict, active_result: dict = None,
               challenge_passive_result: dict = None) -> dict:
        """
        Fuse passive + (optional) active results into a final verdict.

        SYLLABUS: Decision Fusion - the passive CNN/rule-based confidence
        and the active challenge outcome are combined via threshold bands
        into a single ACCEPT / REJECT / PENDING verdict:

          real_confidence >= PASSIVE_ACCEPT_THRESHOLD  -> ACCEPT (no challenge needed)
          real_confidence <= PASSIVE_REJECT_THRESHOLD  -> REJECT (no challenge needed)
          in between                                   -> depends on active_result
            active passed          -> ACCEPT, unless the challenge photo scores
                                       <= CHALLENGE_REJECT_THRESHOLD (see below)
            active failed / None   -> REJECT (or PENDING if not yet run)

        `challenge_passive_result`, if given, is the passive spoof-classifier
        output run on the CHALLENGE photo (not the original photo). Without
        this check, a spoof photo could be shown first (landing in the
        uncertain band) and then swapped for a real live face just to
        satisfy the blink/turn gesture - passing the liveness action alone
        isn't proof the whole attempt was legitimate if the challenge photo
        itself still looks like a print/replay.
        """
        conf = passive_result["real_confidence"]

        if conf >= self.ACCEPT_THRESHOLD:
            return {
                "verdict": "ACCEPT", "confidence": conf,
                "active_used": False, "reason": "high_passive_confidence",
            }

        if conf <= self.REJECT_THRESHOLD:
            return {
                "verdict": "REJECT", "confidence": conf,
                "active_used": False, "reason": "low_passive_confidence",
            }

        # Uncertain band - active challenge required.
        if active_result is None:
            return {
                "verdict": "PENDING", "confidence": conf,
                "active_used": True, "reason": "challenge_required",
            }

        if not active_result["passed"]:
            return {
                "verdict": "REJECT", "confidence": conf,
                "active_used": True,
                "reason": active_result["reason"],
                "challenge": active_result["challenge"],
            }

        if (challenge_passive_result is not None
                and challenge_passive_result["real_confidence"] <= self.CHALLENGE_REJECT_THRESHOLD):
            return {
                "verdict": "REJECT", "confidence": conf,
                "active_used": True,
                "reason": "challenge_photo_looks_spoofed",
                "challenge": active_result["challenge"],
            }

        return {
            "verdict": "ACCEPT", "confidence": conf,
            "active_used": True,
            "reason": active_result["reason"],
            "challenge": active_result["challenge"],
        }
