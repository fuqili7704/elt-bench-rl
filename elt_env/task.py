"""Task specification: what one ELT problem looks like on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# reference/ and ground_truth/ are never copied into the workspace
AGENT_VISIBLE = ("task.yaml", "data_model.yaml", "source", "docs")


@dataclass(frozen=True)
class SourceTable:
    source: str
    type: str       # csv, jsonl, postgres, mongodb, s3, api, file
    table: str
    location: str


@dataclass(frozen=True)
class ELTTask:
    task_id: str
    task_dir: Path
    description: str
    raw_schema: str
    source_tables: tuple[SourceTable, ...]
    model_names: tuple[str, ...]
    # from grading.yaml (official tasks only): raw_row_counts, sort_keys
    grading: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def load(cls, task_dir: str | Path) -> ELTTask:
        task_dir = Path(task_dir).resolve()
        spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
        models = yaml.safe_load((task_dir / "data_model.yaml").read_text(encoding="utf-8"))

        tables = [
            SourceTable(src["name"], src["type"], table, str(task_dir / rel_path))
            for src in spec["sources"]
            for table, rel_path in src["tables"].items()
        ]
        grading_file = task_dir / "grading.yaml"
        grading = yaml.safe_load(grading_file.read_text(encoding="utf-8")) if grading_file.is_file() else {}
        return cls(
            task_id=spec["task_id"],
            task_dir=task_dir,
            description=spec["description"],
            raw_schema=spec["destination"]["raw_schema"],
            source_tables=tuple(tables),
            model_names=tuple(m["name"] for m in models["models"]),
            grading=grading or {},
        )

    def find_source_table(self, source: str, table: str) -> SourceTable | None:
        for t in self.source_tables:
            if t.source == source and t.table == table:
                return t
        return None

    @property
    def reference_dir(self) -> Path:
        return self.task_dir / "reference"
