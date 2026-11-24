"""Run the agent's load_config.yaml.

    destination_schema: raw
    streams:
      - {source: crm, table: customers, destination_table: customers}
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from .sources import read_source_table
from .task import ELTTask
from .utils import check_identifier
from .warehouse import Warehouse

LOAD_CONFIG_FILE = "load_config.yaml"


@dataclass(frozen=True)
class Stream:
    source: str
    table: str
    destination_table: str


@dataclass(frozen=True)
class LoadConfig:
    destination_schema: str
    streams: tuple[Stream, ...]

    @classmethod
    def parse(cls, text: str) -> LoadConfig:
        """Strict parsing with messages the model can act on."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"{LOAD_CONFIG_FILE} is not valid YAML: {e}") from e
        if not isinstance(data, dict) or "streams" not in data or "destination_schema" not in data:
            raise ValueError(f"{LOAD_CONFIG_FILE} needs top-level keys 'destination_schema' and 'streams'")
        streams = []
        for i, s in enumerate(data["streams"] or []):
            missing = {"source", "table", "destination_table"} - set(s or {})
            if missing:
                raise ValueError(f"stream #{i} is missing keys: {sorted(missing)} "
                                 "(each stream needs source, table and destination_table)")
            streams.append(Stream(str(s["source"]), str(s["table"]), check_identifier(str(s["destination_table"]))))
        return cls(check_identifier(str(data["destination_schema"])), tuple(streams))


def run_load(task: ELTTask, config: LoadConfig, warehouse: Warehouse) -> str:
    """Load every stream; report per-stream success so the model gets precise feedback."""
    lines = []
    unknown = False
    for s in config.streams:
        src = task.find_source_table(s.source, s.table)
        if src is None:
            lines.append(f"FAILED source={s.source!r} table={s.table!r}: no such source table")
            unknown = True
            continue
        n = warehouse.write_table(config.destination_schema, s.destination_table, read_source_table(src))
        lines.append(f"loaded {config.destination_schema}.{s.destination_table} ({n} rows)")
    if unknown:
        lines.append("Available (source, table) pairs: "
                     + ", ".join(f"({t.source}, {t.table})" for t in task.source_tables))
    return "\n".join(lines) if lines else "No streams configured."
