# OWNER: Person A | Person B | Person C
# Person A: core/passive_classifier.py, core/spatial_features.py, core/frequency_features.py
# Person B: core/active_challenge.py, core/face_detector.py
# Person C: core/fusion.py, ui/dashboard.py, ui/visualizer.py, utils/logger.py, main.py
"""
SQLite-backed logging of every liveness-check attempt, so the dashboard's
Analytics tab has real history to show.
"""
import sqlite3
import datetime

from utils.constants import DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            verdict TEXT NOT NULL,
            real_confidence REAL,
            texture_var REAL,
            orb_keypoints INTEGER,
            fft_ratio REAL,
            edge_density REAL,
            method TEXT,
            active_used INTEGER,
            challenge_type TEXT,
            challenge_passed INTEGER,
            reason TEXT,
            ear_min REAL,
            yaw_max_delta REAL,
            spoof_type TEXT,
            challenge_confidence REAL
        )
    """)
    # Migrate older DBs (created before ear_min/yaw_max_delta/spoof_type/
    # challenge_confidence existed).
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(attempts)")}
    for col in ("ear_min", "yaw_max_delta", "challenge_confidence"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE attempts ADD COLUMN {col} REAL")
    if "spoof_type" not in existing_cols:
        conn.execute("ALTER TABLE attempts ADD COLUMN spoof_type TEXT")
    conn.commit()
    conn.close()


def log_attempt(passive_result: dict, decision_result: dict, active_result: dict = None,
                 challenge_passive_result: dict = None):
    """
    Persist one liveness-check attempt to SQLite.

    passive_result:            dict returned by PassiveClassifier.predict() for
                                the ORIGINAL photo
    decision_result:           dict returned by DecisionFusion.decide()
    active_result:              optional dict returned by the active challenge runner
    challenge_passive_result:  optional dict - the passive re-check score used by
                                fusion.py's anti-swap check (see CHALLENGE_REJECT_THRESHOLD
                                in utils/constants.py). Logged separately from
                                real_confidence so a REJECT/ACCEPT on the challenge
                                path can be traced back to the actual number that
                                decided it, not just the original photo's score.
    """
    init_db()

    conn = _connect()
    conn.execute(
        """
        INSERT INTO attempts (
            timestamp, verdict, real_confidence, texture_var, orb_keypoints,
            fft_ratio, edge_density, method, active_used, challenge_type,
            challenge_passed, reason, ear_min, yaw_max_delta, spoof_type,
            challenge_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            decision_result.get("verdict"),
            passive_result.get("real_confidence"),
            passive_result.get("texture_var"),
            passive_result.get("orb_keypoints"),
            passive_result.get("fft_ratio"),
            passive_result.get("edge_density"),
            passive_result.get("method"),
            int(bool(decision_result.get("active_used"))),
            (active_result or {}).get("challenge"),
            int(bool((active_result or {}).get("passed"))) if active_result else None,
            decision_result.get("reason"),
            (active_result or {}).get("ear_min"),
            (active_result or {}).get("yaw_max_delta"),
            passive_result.get("spoof_type"),
            (challenge_passive_result or {}).get("real_confidence"),
        ),
    )
    conn.commit()
    conn.close()


def get_recent(n: int = 50) -> list:
    """Return the n most recent attempts as a list of plain dicts (newest first)."""
    init_db()

    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM attempts ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Aggregate counts used by the Analytics tab."""
    init_db()

    conn = _connect()
    total = conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"]
    accept = conn.execute("SELECT COUNT(*) AS c FROM attempts WHERE verdict='ACCEPT'").fetchone()["c"]
    reject = conn.execute("SELECT COUNT(*) AS c FROM attempts WHERE verdict='REJECT'").fetchone()["c"]
    challenge = conn.execute("SELECT COUNT(*) AS c FROM attempts WHERE active_used=1").fetchone()["c"]
    conn.close()

    return {
        "total": total,
        "accept": accept,
        "reject": reject,
        "challenge_rate": (challenge / total) if total else 0.0,
        "accept_rate": (accept / total) if total else 0.0,
        "reject_rate": (reject / total) if total else 0.0,
    }
