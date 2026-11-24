"""Small shared helpers."""

from __future__ import annotations

import re

import pandas as pd

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_identifier(name: str) -> str:
    """Reject schema/table names that could inject SQL (we build DDL with f-strings)."""
    if not _IDENT.match(name):
        raise ValueError(f"invalid identifier {name!r}: use letters, digits and _ only")
    return name


def fetch_limited(con, sql: str, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Run `sql` on a DB-API connection; return at most `max_rows` rows + a truncated flag.

    We never fetch the full result: a `SELECT *` on a big table would blow up
    both memory and the model's context window.
    """
    cur = con.execute(sql)
    if cur.description is None:  # statement without a result set
        return pd.DataFrame(), False
    columns = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows + 1)
    return pd.DataFrame(rows[:max_rows], columns=columns), len(rows) > max_rows


def format_df(df: pd.DataFrame, truncated: bool) -> str:
    if df.columns.empty:
        return "OK (statement returned no result set)"
    text = df.to_string(index=False) if len(df) else f"(0 rows) columns: {list(df.columns)}"
    if truncated:
        text += f"\n... (output truncated to {len(df)} rows)"
    return text


def truncate(text: str, max_chars: int) -> str:
    """Keep the head and the tail; errors usually show up at the end of logs."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [{len(text) - max_chars} chars omitted] ...\n" + text[-half:]
