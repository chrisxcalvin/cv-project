from core.fusion import DecisionFusion
from utils.constants import PASSIVE_ACCEPT_THRESHOLD, PASSIVE_REJECT_THRESHOLD


def _passive(conf):
    return {"real_confidence": conf}


def test_high_confidence_accepts_without_challenge():
    decision = DecisionFusion().decide(_passive(PASSIVE_ACCEPT_THRESHOLD))
    assert decision["verdict"] == "ACCEPT"
    assert decision["active_used"] is False


def test_low_confidence_rejects_without_challenge():
    decision = DecisionFusion().decide(_passive(PASSIVE_REJECT_THRESHOLD))
    assert decision["verdict"] == "REJECT"
    assert decision["active_used"] is False


def test_uncertain_band_pending_when_no_challenge_run_yet():
    mid = (PASSIVE_ACCEPT_THRESHOLD + PASSIVE_REJECT_THRESHOLD) / 2
    decision = DecisionFusion().decide(_passive(mid))
    assert decision["verdict"] == "PENDING"
    assert decision["active_used"] is True


def test_uncertain_band_accepts_when_challenge_passes():
    mid = (PASSIVE_ACCEPT_THRESHOLD + PASSIVE_REJECT_THRESHOLD) / 2
    active_result = {"passed": True, "reason": "blink_confirmed", "challenge": "blink"}
    decision = DecisionFusion().decide(_passive(mid), active_result)
    assert decision["verdict"] == "ACCEPT"


def test_uncertain_band_rejects_when_challenge_fails():
    mid = (PASSIVE_ACCEPT_THRESHOLD + PASSIVE_REJECT_THRESHOLD) / 2
    active_result = {"passed": False, "reason": "timeout", "challenge": "blink"}
    decision = DecisionFusion().decide(_passive(mid), active_result)
    assert decision["verdict"] == "REJECT"


def test_challenge_passed_but_challenge_photo_looks_spoofed_still_rejects():
    """A spoof could pass the blink/turn gesture and then be swapped back -
    the challenge-photo re-check must override an otherwise-passing gesture."""
    mid = (PASSIVE_ACCEPT_THRESHOLD + PASSIVE_REJECT_THRESHOLD) / 2
    active_result = {"passed": True, "reason": "blink_confirmed", "challenge": "blink"}
    challenge_passive = _passive(PASSIVE_REJECT_THRESHOLD)
    decision = DecisionFusion().decide(_passive(mid), active_result, challenge_passive)
    assert decision["verdict"] == "REJECT"
    assert decision["reason"] == "challenge_photo_looks_spoofed"
