"""Snowflake backend. Each rollout gets its own database, dropped at the end.

Names are upper-cased, as Snowflake folds unquoted identifiers.
Settings come from SNOWFLAKE_* environment variables or .env.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import pandas as pd

from ..utils import check_identifier
from .base import Destination, Warehouse

ENV_KEYS = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PRIVATE_KEY_PATH",
            "SNOWFLAKE_ROLE", "SNOWFLAKE_WAREHOUSE")
STATEMENT_TIMEOUT_S = 120

# Agent SQL must be a single read-only statement.
_READ_ONLY = re.compile(r"^\s*(select|with|show|describe|desc)\b", re.IGNORECASE)
_COMMENTS = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


def _q(name: str) -> str:
    """Quoted, upper-cased identifier."""
    return '"' + check_identifier(name).upper() + '"'


def load_env_file(path: str | Path = ".env") -> None:
    """Fill missing os.environ keys from a KEY=VALUE file (no extra dependency)."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class SnowflakeWarehouse(Warehouse):
    name = "snowflake"

    def __init__(self, conn_args: dict, database: str, model_schema: str = "analytics"):
        import snowflake.connector

        self._conn_args = conn_args
        self.database = check_identifier(database).upper()
        self._model_schema = model_schema
        self._lock = threading.Lock()
        self._con = snowflake.connector.connect(**conn_args)
        self._exec(f"CREATE DATABASE IF NOT EXISTS {_q(self.database)}")
        self._exec(f"USE DATABASE {_q(self.database)}")
        self._exec(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {STATEMENT_TIMEOUT_S}")

    @property
    def model_schema(self) -> str:
        return self._model_schema

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._con.cursor()
            cur.execute(sql, params)
            return cur

    def write_table(self, schema: str, table: str, df: pd.DataFrame) -> int:
        from snowflake.connector.pandas_tools import write_pandas

        df = df.rename(columns=lambda c: str(c).upper())
        with self._lock:
            self._con.cursor().execute(f"CREATE SCHEMA IF NOT EXISTS {_q(self.database)}.{_q(schema)}")
            ok, _, nrows, _ = write_pandas(
                self._con, df, table_name=check_identifier(table).upper(),
                database=self.database, schema=check_identifier(schema).upper(),
                auto_create_table=True, overwrite=True, quote_identifiers=True,
                use_logical_type=True,
            )
        if not ok:
            raise RuntimeError(f"write_pandas failed for {schema}.{table}")
        return nrows

    def query(self, sql: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
        """Agent SQL: one read-only statement (multi-statement is off by default in the driver)."""
        if not _READ_ONLY.match(_COMMENTS.sub(" ", sql)):
            raise PermissionError("query_warehouse only runs read-only SELECT / WITH / SHOW / DESCRIBE statements")
        cur = self._exec(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows[:max_rows], columns=cols)
        return df, len(rows) > max_rows

    def fetch_table(self, schema: str, table: str) -> pd.DataFrame | None:
        s, t = check_identifier(schema).upper(), check_identifier(table).upper()
        if (s, t) not in self.list_tables():
            return None
        cur = self._exec(f"SELECT * FROM {_q(self.database)}.{_q(s)}.{_q(t)}")
        return cur.fetch_pandas_all()

    def list_tables(self) -> list[tuple[str, str]]:
        cur = self._exec(
            f"SELECT table_schema, table_name FROM {_q(self.database)}.INFORMATION_SCHEMA.TABLES "
            "WHERE table_schema <> 'INFORMATION_SCHEMA'"
        )
        return [(s, t) for s, t in cur.fetchall()]

    def dbt_profile(self) -> dict:
        a = self._conn_args
        return {
            "type": "snowflake",
            "account": a["account"],
            "user": a["user"],
            "private_key_path": a["private_key_file"],
            "role": a["role"],
            "warehouse": a["warehouse"],
            "database": self.database,
            "schema": self.model_schema,
            "threads": 1,
            "query_tag": "elt_rl",
        }

    def drop(self) -> None:
        try:
            self._exec(f"DROP DATABASE IF EXISTS {_q(self.database)}")
        finally:
            self._con.close()


class SnowflakeDestination(Destination):
    """Creates one database per rollout, all named <prefix>_<namespace>."""

    def __init__(self, account: str, user: str, private_key_path: str, role: str,
                 warehouse: str, prefix: str = "ELT_RL"):
        self.prefix = check_identifier(prefix).upper()
        self.conn_args = {
            "account": account, "user": user, "private_key_file": str(Path(private_key_path).expanduser()),
            "role": role, "warehouse": warehouse,
            "session_parameters": {"QUERY_TAG": "elt_rl"},
            "enable_server_session_keep_alive_auto_detection": False,
        }

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> SnowflakeDestination:
        load_env_file(env_file)
        missing = [k for k in ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"missing Snowflake settings {missing}; set them in the environment or {env_file}")
        e = os.environ
        return cls(e["SNOWFLAKE_ACCOUNT"], e["SNOWFLAKE_USER"], e["SNOWFLAKE_PRIVATE_KEY_PATH"],
                   e["SNOWFLAKE_ROLE"], e["SNOWFLAKE_WAREHOUSE"])

    def create_warehouse(self, namespace: str) -> SnowflakeWarehouse:
        return SnowflakeWarehouse(self.conn_args, f"{self.prefix}_{check_identifier(namespace)}")

    def drop_leftovers(self) -> list[str]:
        """Drop every <prefix>_* database (e.g. after a crashed run). Returns their names."""
        import snowflake.connector

        with snowflake.connector.connect(**self.conn_args) as con:
            cur = con.cursor()
            cur.execute(f"SHOW DATABASES LIKE '{self.prefix}\\\\_%'")
            names = [r[1] for r in cur.fetchall()]
            for n in names:
                cur.execute(f'DROP DATABASE IF EXISTS "{n}"')
        return names
