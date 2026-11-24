"""Official ELT-Bench source types are readable and keep their raw types."""

import pytest

from elt_env.sources import SourceExplorer, read_source_table
from elt_env.task import SourceTable


@pytest.mark.parametrize("kind", ["postgres", "mongodb", "api", "file"])
def test_csv_backed_types(tmp_path, kind):
    p = tmp_path / "t.csv"
    p.write_text("zip_code,city\n501,Holtsville\n544,Holtsville\n")
    df = read_source_table(SourceTable(kind, kind, "t", str(p)))
    assert list(df.columns) == ["zip_code", "city"] and len(df) == 2


def test_s3_jsonl_keeps_strings(tmp_path):
    p = tmp_path / "CBSA.jsonl"
    p.write_text('{"CBSA": "10300", "CBSA_type": "Micro"}\n{"CBSA": "10380", "CBSA_type": "Metro"}\n')
    df = read_source_table(SourceTable("s3", "s3", "CBSA", str(p)))
    assert df["CBSA"].tolist() == ["10300", "10380"]


def test_unknown_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported source type"):
        read_source_table(SourceTable("x", "oracle", "t", str(tmp_path / "t.csv")))


def test_explorer_names_tables_by_source(tmp_path):
    p = tmp_path / "zip_data.csv"
    p.write_text("zip_code,state\n501,NY\n")
    ex = SourceExplorer((SourceTable("postgres", "postgres", "zip_data", str(p)),))
    try:
        df, _ = ex.query("SELECT state FROM postgres.zip_data", 10)
        assert df["state"].tolist() == ["NY"]
    finally:
        ex.close()
