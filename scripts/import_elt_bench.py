"""Convert an official ELT-Bench task into this repo's task layout.

    uv run python scripts/import_elt_bench.py address --elt-bench /path/to/ELT-Bench
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

SOURCE_KINDS = {
    "postgres": "postgres",
    "mongodb": "mongodb",
    "aws_s3": "s3",
    "custom_api": "api",
    "flat_files": "file",
}

DESCRIPTION = """\
This is ELT-Bench task `{task}`. Its data lives in {n_sources} source systems
({sources}); see task.yaml for which tables each one holds. Column descriptions
for every source table are in docs/.
1. Extract & load every source table into the warehouse schema `raw`,
   keeping the source table name as the destination table name.
2. Implement the data models described in data_model.yaml as dbt models.
3. Run the pipeline and submit when you are done.
"""


def _str_presenter(dumper: yaml.SafeDumper, s: str):
    style = "|" if "\n" in s else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)


yaml.SafeDumper.add_representer(str, _str_presenter)


def source_tables(config: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, name in SOURCE_KINDS.items():
        if key not in config:
            continue
        block = config[key]
        if key == "aws_s3":                       # s3:   data: [{path, table}, ...]
            tables = [d["table"] for d in block["data"]]
        elif key == "flat_files":                 # file: [{path, table, format}, ...]
            tables = [d["table"] for d in block]
        else:                                     # postgres / mongodb / api: config.tables
            tables = list(block["config"]["tables"])
        out[name] = tables
    return out


def find_dump(table: str, search_dirs: list[Path]) -> Path:
    for d in search_dirs:
        for ext in (".csv", ".jsonl"):
            if (d / f"{table}{ext}").is_file():
                return d / f"{table}{ext}"
    sys.exit(f"source dump for table {table!r} not found in {[str(d) for d in search_dirs]}. "
             "Unzip data_db.zip / data_api.zip, or download flat files from the URL in config.yaml.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", help="ELT-Bench task name, e.g. address")
    ap.add_argument("--elt-bench", type=Path, required=True, help="path to the ELT-Bench checkout")
    ap.add_argument("--out", type=Path, default=None, help="default: tasks/elt_bench/<task>")
    args = ap.parse_args()

    eb, task = args.elt_bench.resolve(), args.task
    spec_dir = eb / "elt-bench" / "snowflake" / task
    out = (args.out or REPO / "tasks" / "elt_bench" / task).resolve()
    config = yaml.safe_load((spec_dir / "config.yaml").read_text(encoding="utf-8"))
    data_model = yaml.safe_load((spec_dir / "data_model.yaml").read_text(encoding="utf-8"))

    if out.exists():
        shutil.rmtree(out)
    (out / "source").mkdir(parents=True)

    search = [eb / "setup" / "data" / task, eb / "elt-docker" / "rest_api" / "data" / task]
    sources_yaml = []
    for name, tables in source_tables(config).items():
        (out / "source" / name).mkdir()
        mapping = {}
        for t in tables:
            dump = find_dump(t, search)
            shutil.copy2(dump, out / "source" / name / dump.name)
            mapping[t] = f"source/{name}/{dump.name}"
        sources_yaml.append({"name": name, "type": name, "tables": mapping})

    spec = {
        "task_id": task,
        "description": DESCRIPTION.format(task=task, n_sources=len(sources_yaml),
                                          sources=", ".join(s["name"] for s in sources_yaml)),
        "sources": sources_yaml,
        "destination": {"raw_schema": "raw"},
    }
    (out / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    shutil.copy2(spec_dir / "data_model.yaml", out / "data_model.yaml")
    shutil.copytree(eb / "elt-bench" / "schemas" / task, out / "docs")

    gt_src = eb / "ground_truth" / "gt_snowflake" / task
    (out / "ground_truth").mkdir()
    for m in data_model["models"]:
        gt = gt_src / f"{m['name']}.csv"
        if not gt.is_file():
            sys.exit(f"ground truth {gt} not found; download gt_snowflake from Hugging Face first")
        shutil.copy2(gt, out / "ground_truth" / gt.name)

    table_rows = json.loads((eb / "evaluation" / "table.json").read_text(encoding="utf-8")).get(task, {})
    sort_keys = json.loads((eb / "evaluation" / "sort_key.json").read_text(encoding="utf-8")).get(task, {})
    grading = {"raw_row_counts": table_rows, "sort_keys": sort_keys}
    (out / "grading.yaml").write_text(yaml.safe_dump(grading, sort_keys=False), encoding="utf-8")

    n_tables = sum(len(s["tables"]) for s in sources_yaml)
    print(f"imported {task}: {len(sources_yaml)} sources, {n_tables} tables, "
          f"{len(data_model['models'])} models -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


if __name__ == "__main__":
    main()
