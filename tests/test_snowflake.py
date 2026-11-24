"""Credentialed tests on Snowflake (B8). Skipped unless Snowflake settings are configured.

Run only these with:  uv run pytest -q tests/test_snowflake.py
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from elt_env.grader import _score, grade, reference_submission
from elt_env.loader import LoadConfig, Stream, run_load
from elt_env.task import ELTTask
from elt_env.warehouse.snowflake_wh import ENV_KEYS, SnowflakeDestination, load_env_file

ROOT = Path(__file__).resolve().parents[1]
load_env_file(ROOT / ".env")
pytestmark = pytest.mark.skipif(not all(os.environ.get(k) for k in ENV_KEYS),
                                reason="Snowflake not configured (.env)")


@pytest.fixture(scope="module")
def dest():
    return SnowflakeDestination.from_env(ROOT / ".env")


def test_sandbox_is_read_only(dest):
    wh = dest.create_warehouse("test_sandbox")
    try:
        wh.write_table("raw", "t", pd.DataFrame({"x": [1, 2]}))
        assert wh.query("select count(*) as n from raw.t", 5)[0].iloc[0, 0] == 2
        for bad in ["drop table raw.t", "create table raw.u as select 1 as x",
                    "/* hi */ delete from raw.t", "insert into raw.t values (3)"]:
            with pytest.raises(PermissionError):
                wh.query(bad, 5)
        assert len(wh.fetch_table("raw", "t")) == 2
    finally:
        wh.drop()


async def test_toy_shop_reference_scores_one_on_snowflake(dest, tmp_path):
    """Full pipeline on Snowflake: loader -> dbt-snowflake -> grading."""
    task = ELTTask.load(ROOT / "tasks" / "toy_shop")
    r = await grade(task, reference_submission(task), dest, tmp_path)
    assert r.reward == pytest.approx(1.0), r.details


async def test_toy_shop_empty_submission_scores_zero(dest, tmp_path):
    task = ELTTask.load(ROOT / "tasks" / "toy_shop")
    assert (await grade(task, {}, dest, tmp_path)).reward == 0.0


ADDRESS = ROOT / "tasks" / "elt_bench" / "address"


@pytest.mark.skipif(not ADDRESS.is_dir(), reason="run scripts/import_elt_bench.py address first")
def test_address_official_answer_scores_one_on_snowflake(dest):
    """Official task: all 9 source tables loaded into Snowflake match, and the official answer gets 1.0."""
    task = ELTTask.load(ADDRESS)
    wh = dest.create_warehouse("test_address")
    try:
        cfg = LoadConfig(task.raw_schema, tuple(Stream(t.source, t.table, t.table) for t in task.source_tables))
        run_load(task, cfg, wh)
        wh.write_table(wh.model_schema, "states", pd.read_csv(task.task_dir / "ground_truth" / "states.csv"))
        r = _score(task, wh, dbt_ok=True)
        assert r.load_score == 1.0, r.details
        assert r.reward == pytest.approx(1.0) and r.metrics()["official_pass"] == 1.0, r.details
    finally:
        wh.drop()
