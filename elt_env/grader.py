"""Rebuild a submission in a fresh warehouse and score it.

reward = 0.2 * load + 0.8 * models; official_pass follows ELT-Bench's SR_DT.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .official_compare import compare_model, compare_raw
from .loader import LOAD_CONFIG_FILE
from .sources import read_source_table
from .task import ELTTask
from .tools import ELTSession
from .warehouse import Destination, Warehouse
from .workspace import WRITABLE_MODEL_SUFFIXES, Workspace

LOAD_WEIGHT = 0.2
MODEL_WEIGHT = 0.8

Submission = dict[str, str]  # path -> content


@dataclass
class GradeResult:
    reward: float
    load_score: float
    model_scores: dict[str, float]
    details: dict[str, str] = field(default_factory=dict)
    dbt_ok: bool = False

    def metrics(self) -> dict[str, float]:
        m = {"reward": self.reward, "load_score": self.load_score, "dbt_ok": float(self.dbt_ok)}
        m["model_score"] = sum(self.model_scores.values()) / max(len(self.model_scores), 1)
        # SR_DT: models only count if stage 1 fully passed
        m["official_pass"] = m["model_score"] if self.load_score == 1.0 else 0.0
        return m


def _collect(root: Path) -> Submission:
    files: Submission = {}
    if (root / LOAD_CONFIG_FILE).is_file():
        files[LOAD_CONFIG_FILE] = (root / LOAD_CONFIG_FILE).read_text(encoding="utf-8")
    for p in sorted((root / "models").rglob("*")):
        if p.is_file() and p.suffix in WRITABLE_MODEL_SUFFIXES:
            files[p.relative_to(root).as_posix()] = p.read_text(encoding="utf-8")
    return files


def collect_submission(workspace: Workspace) -> Submission:
    return _collect(workspace.root)


def reference_submission(task: ELTTask) -> Submission:
    return _collect(task.reference_dir)


def load_ground_truth(task: ELTTask, model: str) -> pd.DataFrame:
    """Read a ground-truth CSV as text (dtype=str), as the official evaluator does."""
    return pd.read_csv(task.task_dir / "ground_truth" / f"{model}.csv", dtype=str, keep_default_na=True)


def _check_raw_table(task: ELTTask, wh: Warehouse, t) -> tuple[bool, str]:
    """Stage 1 for one source table: present, official row count (if known), identical content."""
    actual = wh.fetch_table(task.raw_schema, t.table)
    if actual is None:
        return False, "table missing"
    expected_rows = task.grading.get("raw_row_counts", {}).get(t.table)
    if expected_rows is not None and len(actual) != expected_rows:
        return False, f"{len(actual)} rows, expected {expected_rows}"
    return compare_raw(actual, read_source_table(t))


def _check_model(task: ELTTask, actual: pd.DataFrame, model: str) -> tuple[bool, str]:
    """Stage 2 for one model. Without official sort keys, rows are sorted by all columns."""
    keys = task.grading.get("sort_keys", {}).get(model, [])
    return compare_model(actual, load_ground_truth(task, model), keys)


async def build_pipeline(task: ELTTask, submission: Submission, destination: Destination,
                         base_dir: str | Path | None = None) -> tuple[ELTSession, bool]:
    """Replay a submission in a brand-new session. Writes go through the same
    whitelist as the agent's, so grading cannot be tricked into running extra files.
    """
    session = ELTSession.create(task, destination, base_dir)
    for path, content in submission.items():
        await session.write_file(path, content)
    await session.run_load()
    dbt_out = await session.run_dbt()
    return session, dbt_out.startswith("dbt run succeeded")


def _score(task: ELTTask, wh: Warehouse, dbt_ok: bool) -> GradeResult:
    """Score the rebuilt warehouse table by table."""
    details: dict[str, str] = {}

    # Stage 1: every source table must land in raw_schema.
    loaded = 0
    for t in task.source_tables:
        ok, why = _check_raw_table(task, wh, t)
        loaded += ok
        if not ok:
            details[f"{task.raw_schema}.{t.table}"] = why
    load_score = loaded / len(task.source_tables)

    # Stage 2: each data model is all-or-nothing (no partial credit for "some right rows").
    model_scores: dict[str, float] = {}
    for m in task.model_names:
        actual = wh.fetch_table(wh.model_schema, m)
        ok, why = (False, "model missing") if actual is None else _check_model(task, actual, m)
        model_scores[m] = float(ok)
        if not ok:
            details[m] = why
    model_score = sum(model_scores.values()) / len(model_scores)

    reward = LOAD_WEIGHT * load_score + MODEL_WEIGHT * model_score
    return GradeResult(reward, load_score, model_scores, details, dbt_ok)


async def grade(task: ELTTask, submission: Submission, destination: Destination,
                base_dir: str | Path | None = None) -> GradeResult:
    session, dbt_ok = await build_pipeline(task, submission, destination, base_dir)
    try:
        return await asyncio.to_thread(_score, task, session.warehouse, dbt_ok)
    finally:
        await session.close()
