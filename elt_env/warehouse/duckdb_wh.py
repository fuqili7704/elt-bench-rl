"""DuckDB destination: a local, credential-free stand-in for Snowflake.

Isolation is trivial: one rollout = one .duckdb file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from ..utils import check_identifier, fetch_limited
from .base import Destination, Warehouse


class DuckDBWarehouse(Warehouse):
    name = "duckdb"

    def __init__(self, path: Path, model_schema: str = "analytics"):
        self.path = Path(path)
        self._model_schema = model_schema
        duckdb.connect(str(self.path)).close()

    @property
    def model_schema(self) -> str:
        return self._model_schema

    # short-lived connections: dbt needs the file lock in its own process

    def write_table(self, schema: str, table: str, df: pd.DataFrame) -> int:
        schema, table = check_identifier(schema), check_identifier(table)
        with duckdb.connect(str(self.path)) as con:
            con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            con.register("_incoming", df)
            con.execute(f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS SELECT * FROM _incoming')
        return len(df)

    def query(self, sql: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            return fetch_limited(con, sql, max_rows)

    def fetch_table(self, schema: str, table: str) -> pd.DataFrame | None:
        if (schema, table) not in self.list_tables():
            return None
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(f'SELECT * FROM "{schema}"."{table}"').df()

    def list_tables(self) -> list[tuple[str, str]]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_catalog = current_database()"
            ).fetchall()
        return [(s, t) for s, t in rows]

    def dbt_profile(self) -> dict:
        return {
            "type": "duckdb",
            "path": str(self.path),
            "schema": self.model_schema,
            "threads": 1,
            # Agent-written models run with file access disabled, too.
            "settings": {"enable_external_access": False},
        }

    def drop(self) -> None:
        for p in (self.path, self.path.with_suffix(".duckdb.wal")):
            p.unlink(missing_ok=True)


class DuckDBDestination(Destination):

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or Path(tempfile.gettempdir()) / "elt_warehouses")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_warehouse(self, namespace: str) -> DuckDBWarehouse:
        return DuckDBWarehouse(self.base_dir / f"{check_identifier(namespace)}.duckdb")
