"""Grader checks on the official address task, using its ground truth and broken copies of it."""

from pathlib import Path

import pandas as pd
import pytest

from elt_env.grader import _score
from elt_env.loader import LoadConfig, Stream, run_load
from elt_env.task import ELTTask
from elt_env.warehouse import DuckDBDestination

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "elt_bench" / "address"
pytestmark = pytest.mark.skipif(not TASK_DIR.is_dir(), reason="run scripts/import_elt_bench.py address first")


def full_load_config(task: ELTTask) -> LoadConfig:
    return LoadConfig(task.raw_schema, tuple(Stream(t.source, t.table, t.table) for t in task.source_tables))


@pytest.fixture(scope="module")
def task():
    return ELTTask.load(TASK_DIR)


@pytest.fixture(scope="module")
def wh(task, tmp_path_factory):
    w = DuckDBDestination(tmp_path_factory.mktemp("wh")).create_warehouse("address")
    run_load(task, full_load_config(task), w)
    yield w
    w.drop()


@pytest.fixture
def gold(task):
    return pd.read_csv(task.task_dir / "ground_truth" / "states.csv")


def submit_model(wh, df):
    wh.write_table(wh.model_schema, "states", df)


def test_official_answer_scores_one(task, wh, gold):
    submit_model(wh, gold)
    r = _score(task, wh, dbt_ok=True)
    assert r.load_score == 1.0, r.details
    assert r.reward == pytest.approx(1.0), r.details
    assert r.metrics()["official_pass"] == 1.0


def test_harmless_format_differences_still_score_one(task, wh, gold):
    df = gold.sample(frac=1, random_state=0).rename(columns=str.lower)
    df["name"] = df["name"].str.upper()
    df["extra"] = 1
    submit_model(wh, df)
    assert _score(task, wh, dbt_ok=True).reward == pytest.approx(1.0)


def test_no_model_scores_load_only(task, wh):
    import duckdb
    with duckdb.connect(str(wh.path)) as con:
        con.execute(f'DROP TABLE IF EXISTS "{wh.model_schema}"."states"')
    r = _score(task, wh, dbt_ok=False)
    assert r.reward == pytest.approx(0.2) and r.metrics()["official_pass"] == 0.0


@pytest.mark.parametrize("corrupt", [
    "count_plus_one", "flip_flag", "null_instead_of_zero", "drop_a_state", "wrong_county",
])
def test_single_corruption_is_caught(task, wh, gold, corrupt):
    df = gold.copy()
    if corrupt == "count_plus_one":
        df.loc[0, "NUM_COUNTIES"] += 1
    elif corrupt == "flip_flag":
        df.loc[0, "HIGHER_AVG_FEMALE_PER"] = 1 - df.loc[0, "HIGHER_AVG_FEMALE_PER"]
    elif corrupt == "null_instead_of_zero":
        zero = df.index[df["NUM_ASIAN"] == 0]
        df["NUM_ASIAN"] = df["NUM_ASIAN"].astype("float")
        df.loc[zero[0] if len(zero) else 0, "NUM_ASIAN"] = None
    elif corrupt == "drop_a_state":
        df = df.iloc[1:]
    elif corrupt == "wrong_county":
        df.loc[0, "COUNTY_WITH_HIGHEST_AVERAGE_INCOME_PER_HOUSEHOLD"] = "NOT A COUNTY"
    submit_model(wh, df)
    r = _score(task, wh, dbt_ok=True)
    assert r.model_scores == {"states": 0.0}, r.details
    assert r.reward == pytest.approx(0.2)


def test_wrong_table_mapping_fails_stage1(task, gold, tmp_path):
    w = DuckDBDestination(tmp_path).create_warehouse("bad_load")
    try:
        streams = [Stream(t.source, t.table, t.table) for t in task.source_tables if t.table != "alias"]
        streams.append(Stream("api", "avoid", "alias"))
        run_load(task, LoadConfig(task.raw_schema, tuple(streams)), w)
        w.write_table(w.model_schema, "states", gold)
        r = _score(task, w, dbt_ok=True)
        assert r.load_score == pytest.approx(8 / 9), r.details
        assert "raw.alias" in r.details
        assert r.model_scores == {"states": 1.0}
        assert r.metrics()["official_pass"] == 0.0
    finally:
        w.drop()
