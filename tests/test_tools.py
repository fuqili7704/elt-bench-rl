"""Credential-free tests for the tool layer, using the toy_shop task.

A "scripted agent" replays the reference solution through the tools, then we
try the cheats we know about and check they are blocked.
"""

from pathlib import Path

import pytest

from elt_env.task import ELTTask
from elt_env.tools import ELTSession
from elt_env.warehouse import DuckDBDestination

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "toy_shop"
SECRET = (TASK_DIR / "reference" / "models" / "city_sales.sql").as_posix()

LOAD_CONFIG = """\
destination_schema: raw
streams:
  - {source: crm, table: customers, destination_table: customers}
  - {source: crm, table: products, destination_table: products}
  - {source: orders_store, table: orders, destination_table: orders}
"""


@pytest.fixture
async def session(tmp_path):
    s = ELTSession.create(ELTTask.load(TASK_DIR), DuckDBDestination(tmp_path / "wh"), base_dir=tmp_path)
    yield s
    await s.close()


async def load_and_write_reference(s: ELTSession) -> None:
    await s.write_file("load_config.yaml", LOAD_CONFIG)
    await s.run_load()
    for f in (TASK_DIR / "reference" / "models").iterdir():
        await s.write_file(f"models/{f.name}", f.read_text())


async def test_reference_pipeline_end_to_end(session):
    files = await session.list_files()
    assert "task.yaml" in files and "data_model.yaml" in files

    assert "Alice" in await session.query_source("SELECT * FROM crm.customers")

    await session.write_file("load_config.yaml", LOAD_CONFIG)
    out = await session.run_load()
    assert "loaded raw.orders (10 rows)" in out, out

    await load_and_write_reference(session)
    out = await session.run_dbt()
    assert out.startswith("dbt run succeeded"), out

    out = await session.query_warehouse(
        "SELECT order_count, total_revenue FROM analytics.customer_revenue WHERE customer_name = 'Frank'"
    )
    assert "0" in out, out  # Frank has no orders but must still appear


async def test_dbt_error_is_returned_as_feedback(session):
    await session.write_file("models/broken.sql", "SELECT no_such_column FROM nowhere")
    out = await session.run_dbt()
    assert out.startswith("dbt run FAILED"), out
    assert session.tool_errors["run_dbt"] == 0  # a failed run is feedback, not a tool crash


async def test_reference_is_not_in_workspace(session):
    assert "reference" not in await session.list_files()


@pytest.mark.parametrize("path", ["../reference/models/city_sales.sql", SECRET, "../../../../etc/passwd"])
async def test_file_tools_cannot_escape_workspace(session, path):
    assert (await session.read_file(path)).startswith("ERROR")


@pytest.mark.parametrize("path", ["dbt_project.yml", "models/evil.py", "macros/x.sql", "task.yaml"])
async def test_only_whitelisted_files_are_writable(session, path):
    assert (await session.write_file(path, "x")).startswith("ERROR")


async def test_sql_cannot_read_files_via_query_source(session):
    out = await session.query_source(f"SELECT * FROM read_text('{SECRET}')")
    assert out.startswith("ERROR"), out


async def test_sql_cannot_read_files_via_query_warehouse(session):
    out = await session.query_warehouse(f"SELECT * FROM read_text('{SECRET}')")
    assert out.startswith("ERROR"), out


async def test_sql_cannot_reenable_file_access(session):
    out = await session.query_source("SET enable_external_access = true")
    assert out.startswith("ERROR"), out


async def test_query_warehouse_is_read_only(session):
    await load_and_write_reference(session)
    out = await session.query_warehouse("CREATE TABLE analytics.city_sales AS SELECT 1 AS x")
    assert out.startswith("ERROR"), out


async def test_dbt_models_cannot_read_files(session):
    await load_and_write_reference(session)
    await session.write_file("models/peek.sql", f"SELECT * FROM read_text('{SECRET}')")
    out = await session.run_dbt()
    assert out.startswith("dbt run FAILED"), out
