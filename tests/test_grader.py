"""End-to-end grading on toy_shop. Requires official_compare.py and ground_truth/ to exist."""

from pathlib import Path

import pytest

from elt_env.grader import grade, reference_submission
from elt_env.task import ELTTask
from elt_env.tools import ELTSession
from elt_env.warehouse import DuckDBDestination

TASK = ELTTask.load(Path(__file__).resolve().parents[1] / "tasks" / "toy_shop")
REF = reference_submission(TASK)

# Classic mistake: INNER JOIN drops Frank (no orders) -> customer_revenue wrong,
# and city_sales (built on top of it) is wrong too.
INNER_JOIN_BUG = REF["models/customer_revenue.sql"].replace("left join", "join")


@pytest.fixture
def dest(tmp_path):
    return DuckDBDestination(tmp_path / "wh")


async def test_reference_scores_full_reward(dest, tmp_path):
    r = await grade(TASK, REF, dest, tmp_path)
    assert r.reward == pytest.approx(1.0), r.details


async def test_empty_submission_scores_zero(dest, tmp_path):
    r = await grade(TASK, {}, dest, tmp_path)
    assert r.reward == 0.0


async def test_load_only_gets_stage1_credit(dest, tmp_path):
    r = await grade(TASK, {"load_config.yaml": REF["load_config.yaml"]}, dest, tmp_path)
    assert r.load_score == 1.0 and r.reward == pytest.approx(0.2)


async def test_one_of_two_models_gets_partial_credit(dest, tmp_path):
    sub = {k: v for k, v in REF.items() if "city_sales" not in k}
    r = await grade(TASK, sub, dest, tmp_path)
    assert r.model_scores == {"customer_revenue": 1.0, "city_sales": 0.0}
    assert r.reward == pytest.approx(0.2 + 0.8 * 0.5)


async def test_inner_join_bug_is_caught(dest, tmp_path):
    r = await grade(TASK, {**REF, "models/customer_revenue.sql": INNER_JOIN_BUG}, dest, tmp_path)
    assert r.model_scores == {"customer_revenue": 0.0, "city_sales": 0.0}
    assert r.reward == pytest.approx(0.2)


async def test_manual_tables_do_not_count(dest, tmp_path):
    """The agent's own warehouse is irrelevant: grading rebuilds from the submission only."""
    s = ELTSession.create(TASK, dest, tmp_path)
    try:
        for path, content in REF.items():
            await s.write_file(path, content)
        await s.run_load()
        assert (await s.run_dbt()).startswith("dbt run succeeded")
        # ...but the agent then deletes its model files before submitting
        r = await grade(TASK, {"load_config.yaml": REF["load_config.yaml"]}, dest, tmp_path)
        assert r.model_scores == {"customer_revenue": 0.0, "city_sales": 0.0}
    finally:
        await s.close()


async def test_ground_truth_is_hidden(dest, tmp_path):
    s = ELTSession.create(TASK, dest, tmp_path)
    try:
        assert "ground_truth" not in await s.list_files()
    finally:
        await s.close()
