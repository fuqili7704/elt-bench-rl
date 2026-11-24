"""Spec for the official comparison rules: each test is one rule.

The table and most cases come from the Phase-A test_compare.py; where the
official stages disagree, the same input is tested against both.
"""

import io

import pandas as pd
import pytest

from elt_env.official_compare import compare_model, compare_raw

EXPECTED = pd.DataFrame({
    "city": ["Chicago", "Urbana", "Springfield"],
    "num_customers": [3, 2, 1],
    "total_revenue": [64.49, 64.73, 18.00],
})


def as_gold(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)


def raw(actual, expected=EXPECTED) -> bool:
    ok, why = compare_raw(actual, expected)
    assert ok or why, "a failed comparison must explain why"
    return ok


def model(actual, expected=EXPECTED, keys=("city",)) -> bool:
    ok, why = compare_model(actual, as_gold(expected), list(keys))
    assert ok or why, "a failed comparison must explain why"
    return ok


both = pytest.mark.parametrize("check", [raw, model], ids=["stage1", "stage2"])


@both
def test_identical(check):
    assert check(EXPECTED.copy())


@both
def test_row_order_does_not_matter(check):
    assert check(EXPECTED.iloc[[2, 0, 1]])


@both
def test_column_order_does_not_matter(check):
    assert check(EXPECTED[["total_revenue", "city", "num_customers"]])


@both
def test_column_names_case_insensitive(check):   # Snowflake upper-cases names
    assert check(EXPECTED.rename(columns=str.upper))


@both
def test_int_equals_float(check):
    assert check(EXPECTED.assign(num_customers=[3.0, 2.0, 1.0]))


@both
def test_null_equals_null(check):
    exp = pd.DataFrame({"name": ["Frank", "Alice"], "x": [None, 1.0]})
    assert check(exp.copy(), exp) if check is raw else check(exp.copy(), exp, keys=("name",))


@both
def test_missing_row(check):
    assert not check(EXPECTED.iloc[:2])


@both
def test_extra_row(check):
    extra = pd.DataFrame({"city": ["Boston"], "num_customers": [1], "total_revenue": [5.0]})
    assert not check(pd.concat([EXPECTED, extra]))


@both
def test_wrong_value(check):
    assert not check(EXPECTED.assign(total_revenue=[64.49, 64.73, 99.0]))


@both
def test_missing_column(check):
    assert not check(EXPECTED.drop(columns=["num_customers"]))


@both
def test_wrong_column_name(check):               # num_of_customers != num_customers
    assert not check(EXPECTED.rename(columns={"num_customers": "num_of_customers"}))


@both
@pytest.mark.parametrize("value", [0, 0.0])
def test_null_is_not_zero(check, value):         # Frank's revenue NULL vs 0
    exp = pd.DataFrame({"name": ["Frank"], "revenue": [None]})
    act = pd.DataFrame({"name": ["Frank"], "revenue": [value]})
    assert not (check(act, exp) if check is raw else check(act, exp, keys=("name",)))


def test_duplicates_count_stage1():              # multiset, not set
    assert not raw(pd.DataFrame({"k": ["A", "B", "B"]}), pd.DataFrame({"k": ["A", "A", "B"]}))


def test_duplicates_count_stage2():
    assert not model(pd.DataFrame({"k": ["A", "B", "B"]}), pd.DataFrame({"k": ["A", "A", "B"]}), keys=("k",))


def test_extra_columns_are_ignored_stage2():
    assert model(EXPECTED.assign(avg_revenue=[1.0, 2.0, 3.0]))


def test_text_case():
    act = EXPECTED.assign(city=["Chicago", "urbana", "Springfield"])
    assert not raw(act)
    assert model(act)


def test_surrounding_spaces():
    act = EXPECTED.assign(city=["Chicago ", "Urbana", "Springfield"])
    assert not raw(act)
    assert model(act)


def test_float_noise():
    act = EXPECTED.assign(total_revenue=[64.4899999, 64.73, 18.0000001])
    assert not raw(act)
    assert model(act)


def test_one_unit_in_last_shared_decimal():
    act = EXPECTED.assign(total_revenue=[64.495, 64.73, 18.0])
    assert not raw(act)
    assert model(act)


def test_beyond_last_shared_decimal_fails_stage2():
    assert not model(EXPECTED.assign(total_revenue=[64.52, 64.73, 18.0]))


def test_extra_decimals_are_fine_stage2():
    assert model(EXPECTED.assign(total_revenue=[64.4912, 64.73, 18.0]))


def test_integer_column_must_be_exact_stage2():
    assert not model(EXPECTED.assign(num_customers=[3.4, 2, 1]))


def test_loader_metadata_columns_are_ignored_stage1():
    act = EXPECTED.assign(_AIRBYTE_RAW_ID=["a", "b", "c"], _AB_SOURCE_FILE_URL=["x"] * 3, _ID=[1, 2, 3])
    assert raw(act)


def test_numeric_strings_stay_text_stage1():
    src = pd.DataFrame({"CBSA": ["10300", "10380"]})
    assert raw(src.copy(), src)
    assert not raw(pd.DataFrame({"CBSA": ["10300.0", "10380.0"]}), src)
