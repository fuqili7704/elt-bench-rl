"""Source connectors: read one source table into a DataFrame.

Adding a new source type (postgres, mongodb, ...) = register one reader function.
Nothing else in the environment changes.
"""

from __future__ import annotations

from typing import Callable

import duckdb
import pandas as pd

from .task import SourceTable
from .utils import check_identifier, fetch_limited

SourceReader = Callable[[SourceTable], pd.DataFrame]

_READERS: dict[str, SourceReader] = {}


def register_reader(type_name: str):
    def deco(fn: SourceReader) -> SourceReader:
        _READERS[type_name] = fn
        return fn
    return deco


@register_reader("csv")
def _read_csv(t: SourceTable) -> pd.DataFrame:
    return pd.read_csv(t.location)


@register_reader("jsonl")
def _read_jsonl(t: SourceTable) -> pd.DataFrame:
    return pd.read_json(t.location, lines=True)


# Official source types read ELT-Bench's CSV/JSONL dumps directly instead of live services.

def _read_dump(t: SourceTable) -> pd.DataFrame:
    """Read a source dump by file extension."""
    if t.location.endswith(".jsonl"):
        # keep "10300" as a string, like the source
        return pd.read_json(t.location, lines=True, dtype=False)
    return pd.read_csv(t.location, low_memory=False)


for _type in ("postgres", "mongodb", "s3", "api", "file"):
    register_reader(_type)(_read_dump)


def read_source_table(t: SourceTable) -> pd.DataFrame:
    if t.type not in _READERS:
        raise ValueError(f"unsupported source type {t.type!r}; known: {sorted(_READERS)}")
    return _READERS[t.type](t)


class SourceExplorer:
    """Sandboxed, read-mostly SQL access to the task's source data (tool: query_source).

    Every source table is copied into an in-memory DuckDB as <source>.<table>,
    then file access is switched off, so `read_text('/any/path')` cannot be
    used to peek at the reference solution.
    """

    def __init__(self, tables: tuple[SourceTable, ...]):
        self.con = duckdb.connect()
        for t in tables:
            src, tbl = check_identifier(t.source), check_identifier(t.table)
            df = read_source_table(t)
            self.con.execute(f'CREATE SCHEMA IF NOT EXISTS "{src}"')
            self.con.register("_incoming", df)
            self.con.execute(f'CREATE TABLE "{src}"."{tbl}" AS SELECT * FROM _incoming')
            self.con.unregister("_incoming")
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")

    def query(self, sql: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
        return fetch_limited(self.con, sql, max_rows)

    def close(self) -> None:
        self.con.close()
