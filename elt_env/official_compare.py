"""ELT-Bench's comparison rules.

compare_raw follows evaluation/eva_stage1_stored.py, compare_model follows
evaluation/eva_stage2.py (https://github.com/uiuc-kang-lab/ELT-Bench).
"""

from __future__ import annotations

import io
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

_NULL = "\x00NULL"


def _is_loader_metadata(col: str) -> bool:
    """Columns the official check ignores: Airbyte _AIRBYTE_* / _AB_* and MongoDB _ID."""
    c = col.upper()
    return c.startswith("_AIRBYTE") or c.startswith("_AB_") or c == "_ID"


def _canon(v) -> str:
    """One value -> canonical text, so that equal values render identically."""
    if v is None or (not isinstance(v, (str, bytes, list, dict)) and pd.isna(v)):
        return _NULL
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, Decimal):
        if v == v.to_integral_value():
            return str(int(v))
        v = float(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v.is_integer() and abs(v) < 2 ** 53:
            return str(int(v))
        return format(v, ".15g")
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def compare_raw(actual: pd.DataFrame, source: pd.DataFrame) -> tuple[bool, str]:
    """Return (is_match, reason) for one loaded raw table vs its source data."""
    actual = actual[[c for c in actual.columns if not _is_loader_metadata(str(c))]]
    lower = {str(c).lower(): c for c in actual.columns}
    cols = [str(c) for c in source.columns]
    missing = [c for c in cols if c.lower() not in lower]
    if missing:
        return False, f"missing columns: {missing}"
    if len(actual) != len(source):
        return False, f"{len(actual)} rows, expected {len(source)}"

    a = pd.DataFrame({c: actual[lower[c.lower()]].map(_canon).to_numpy() for c in cols})
    s = pd.DataFrame({c: source[c].map(_canon).to_numpy() for c in cols})
    if Counter(map(tuple, a.to_numpy())) == Counter(map(tuple, s.to_numpy())):
        return True, ""
    bad = [c for c in cols if Counter(a[c]) != Counter(s[c])]
    return False, f"content differs in columns: {bad or 'row combinations'}"


RTOL, ATOL, NOISE, MIN_DECIMALS = 1e-2, 1e-9, 1e-9, 1
_INT_LITERAL = re.compile(r"[-+]?\d+(\.0*)?")


def as_csv_text(df: pd.DataFrame) -> pd.DataFrame:
    """Round-trip a table through CSV text, as the official evaluator does."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, keep_default_na=True)


def _sort_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    n = col.notna().sum()
    if n > 0 and int(num.notna().sum()) == int(n):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort by the unique key, then by every other column as a deterministic tie-breaker."""
    if df.empty:
        return df.reset_index(drop=True)
    lower = {c.lower(): c for c in df.columns}
    resolved: list[str] = []
    for k in keys:
        c = lower.get(k.lower())
        if c is not None and c not in resolved:
            resolved.append(c)
    cols = resolved + [c for c in df.columns if c not in resolved]
    aux = pd.DataFrame({c: _sort_key(df[c]) for c in cols})
    order = aux.sort_values(by=cols, kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced if coerced.isna().sum() == s.isna().sum() else s


def _text_match(v1: pd.Series, v2: pd.Series) -> bool:
    """Trimmed, case-insensitive text equality; NULL only matches NULL.

    Vectorized version of the official per-row loop (same result, much faster).
    """
    n1, n2 = v1.isna().to_numpy(), v2.isna().to_numpy()
    if np.any(n1 != n2):
        return False
    t1 = v1[~n1].astype(str).str.strip().str.lower().to_numpy()
    t2 = v2[~n2].astype(str).str.strip().str.lower().to_numpy()
    return bool(np.all(t1 == t2))


def _decimals(v: str) -> int:
    try:
        e = Decimal(v.strip()).normalize().as_tuple().exponent
    except (InvalidOperation, ValueError):
        return 0
    return max(0, -e) if isinstance(e, int) else 0


def _column_decimals(s: pd.Series) -> int:
    s = s.dropna().astype(str).str.strip()
    if s.empty:
        return 0
    plain = s.str.fullmatch(r"[-+]?\d*\.?\d*")
    # partition() instead of the official split()+fillna(""): same result, no pandas warning.
    frac = s[plain].str.partition(".")[2].str.rstrip("0").str.len()
    m = int(frac.max()) if len(frac) else 0
    for v in s[~plain].unique():
        m = max(m, _decimals(v))
    return m


def _columns_match(gold: pd.Series, out: pd.Series) -> bool:
    """One gold column vs one output column, both already sorted the same way."""
    if len(gold) != len(out):
        return False
    gold, out = gold.reset_index(drop=True), out.reset_index(drop=True)
    n1, n2 = _to_numeric_if_possible(gold), _to_numeric_if_possible(out)
    if not (pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2)):
        return _text_match(gold, out)

    a, b = n1.to_numpy(dtype=float), n2.to_numpy(dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if np.any(nan_a != nan_b):
        return False
    keep = ~nan_a
    a, b = a[keep], b[keep]
    if not np.all(np.isclose(a, b, rtol=RTOL, atol=ATOL)):
        return False
    diff = np.abs(a - b)

    if _column_decimals(gold) == 0:
        ok = (diff <= ATOL + NOISE * np.abs(b)) & (diff < 0.5)
        gs = gold[keep].astype(str).str.strip().to_numpy()
        os_ = out[keep].astype(str).str.strip().to_numpy()
        for i in np.where((np.abs(a) >= 2 ** 53) | (np.abs(b) >= 2 ** 53))[0]:
            if _INT_LITERAL.fullmatch(gs[i]) and _INT_LITERAL.fullmatch(os_[i]):
                ok[i] = Decimal(gs[i]) == Decimal(os_[i])
            elif _INT_LITERAL.fullmatch(gs[i]):
                ok[i] = a[i] == b[i]
            else:
                ok[i] = diff[i] <= ATOL + NOISE * abs(b[i])
        return bool(np.all(ok))

    places = max(min(_column_decimals(gold), _column_decimals(out)), MIN_DECIMALS)
    return bool(np.all(diff <= 10.0 ** -places * 1.000001 + ATOL + NOISE * np.abs(b)))


def compare_model(actual: pd.DataFrame, expected_csv_text: pd.DataFrame,
                  keys: list[str]) -> tuple[bool, str]:
    """Return (is_match, reason) for one model table vs its ground truth.

    `expected_csv_text` must be read with dtype=str (as the official script does);
    `keys` are the table's unique-key columns from sort_key.json (may be empty).
    """
    out = sort_by_keys(as_csv_text(actual), keys)
    gold = sort_by_keys(expected_csv_text, keys)
    lower = {c.lower(): c for c in out.columns}
    missed = [g for g in gold.columns if g.lower() not in lower]
    if missed:
        return False, f"missing columns: {missed}"
    if len(out) != len(gold):
        return False, f"row count {len(out)} != expected {len(gold)}"
    bad = [g for g in gold.columns if not _columns_match(gold[g], out[lower[g.lower()]])]
    if bad:
        return False, f"columns differ: {bad}"
    return True, ""
