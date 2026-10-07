"""Durable, incremental metrics store for preference-steering experiments.

A single SQLite database (default: <results_dir>/metrics.sqlite) records EVERY
example and EVERY steering step as it happens -- not just final aggregates -- so
multi-hour runs are crash-resilient and fully analyzable afterwards. Rows are
committed as they are produced; if a run dies at hour 3, everything up to that
point is preserved.

Schema
------
runs      one row per experiment invocation (run_id + full config snapshot)
examples  one row per generated/evaluated example (per user, per round, per mode)
steps     one row per diffusion denoising step that ran steering (the fine grain)

Vectors (features, theta, u_A, true preference) are stored as JSON text so the
same schema works for any feature dimension (3-d concepts or 512-d embeddings).

Analyze later, e.g.:
    import sqlite3, pandas as pd
    con = sqlite3.connect("preference/results/metrics.sqlite")
    ex = pd.read_sql("SELECT * FROM examples", con)
    ex.groupby(["feature_space","mode","round"]).clicked.mean()  # convergence curves
    steps = pd.read_sql("SELECT * FROM steps", con)               # per-step steering
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    created_utc       TEXT,
    base_model        TEXT,
    lora              TEXT,
    use_real_diffusion INTEGER,
    git_commit        TEXT,
    config_json       TEXT
);

CREATE TABLE IF NOT EXISTS examples (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    feature_space     TEXT,
    mode              TEXT,     -- fixed_theta | adaptive_theta
    phase             TEXT,     -- bootstrap | generation
    user_id           INTEGER,
    round             INTEGER,
    clicked           INTEGER,
    pref_score        REAL,     -- (theta + u_A) . f
    theta_dot_f       REAL,
    uA_dot_f          REAL,
    alignment_uA_true REAL,     -- cos(u_A, true_preference)
    cum_click_rate    REAL,
    image_path        TEXT,
    image_id          TEXT,
    f_json            TEXT,     -- generated/example feature vector
    uA_json           TEXT,     -- learned user vector at this round
    theta_json        TEXT,     -- shared theta at this round
    true_pref_json    TEXT,
    created_utc       TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    feature_space     TEXT,
    mode              TEXT,
    user_id           INTEGER,
    round             INTEGER,
    step_idx          INTEGER,
    timestep          REAL,
    pref_score        REAL,
    grad_norm         REAL,
    steered           INTEGER
);

CREATE INDEX IF NOT EXISTS idx_examples_run ON examples(run_id, feature_space, mode);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, feature_space, mode, user_id, round);
"""


def _utc() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _jsonify(v):
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    return json.dumps(v)


class MetricsDB:
    """Append-only SQLite logger. One instance per experiment process."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.run_id: str | None = None

    def start_run(self, config: dict | None = None, **meta) -> str:
        """Register a run; returns its run_id (UTC timestamp)."""
        self.run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
        self.conn.execute(
            "INSERT INTO runs (run_id, created_utc, base_model, lora, "
            "use_real_diffusion, git_commit, config_json) VALUES (?,?,?,?,?,?,?)",
            (
                self.run_id,
                _utc(),
                meta.get("base_model"),
                meta.get("lora"),
                int(bool(meta.get("use_real_diffusion"))),
                meta.get("git_commit"),
                json.dumps(config or {}, default=str),
            ),
        )
        self.conn.commit()
        return self.run_id

    def log_example(
        self,
        *,
        feature_space: str,
        mode: str,
        phase: str,
        user_id: int,
        round: int,
        clicked: bool,
        pref_score: float = None,
        theta_dot_f: float = None,
        uA_dot_f: float = None,
        alignment_uA_true: float = None,
        cum_click_rate: float = None,
        image_path: str = None,
        image_id: str = None,
        f=None,
        uA=None,
        theta=None,
        true_pref=None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO examples (run_id, feature_space, mode, phase, user_id, "
            "round, clicked, pref_score, theta_dot_f, uA_dot_f, alignment_uA_true, "
            "cum_click_rate, image_path, image_id, f_json, uA_json, theta_json, "
            "true_pref_json, created_utc) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id, feature_space, mode, phase, user_id, round,
                int(bool(clicked)),
                _f(pref_score), _f(theta_dot_f), _f(uA_dot_f), _f(alignment_uA_true),
                _f(cum_click_rate), image_path, image_id,
                _jsonify(f), _jsonify(uA), _jsonify(theta), _jsonify(true_pref),
                _utc(),
            ),
        )
        self.conn.commit()  # durable per example

    def log_step(
        self,
        *,
        feature_space: str,
        mode: str,
        user_id: int,
        round: int,
        step_idx: int,
        timestep: float = None,
        pref_score: float = None,
        grad_norm: float = None,
        steered: bool = False,
    ) -> None:
        # No commit here; flushed by the next log_example (one transaction/round).
        self.conn.execute(
            "INSERT INTO steps (run_id, feature_space, mode, user_id, round, "
            "step_idx, timestep, pref_score, grad_norm, steered) VALUES "
            "(?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id, feature_space, mode, user_id, round, step_idx,
                _f(timestep), _f(pref_score), _f(grad_norm), int(bool(steered)),
            ),
        )

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()


def _f(v):
    """Coerce tensors/np scalars to plain float (or None)."""
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:
            pass
    return float(v)
