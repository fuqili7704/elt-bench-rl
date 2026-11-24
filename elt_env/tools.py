"""Agent tools for one rollout. Errors are returned as text so the model can react to them."""

from __future__ import annotations

import asyncio
import functools
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .dbt_runner import run_dbt_project, write_profiles
from .loader import LOAD_CONFIG_FILE, LoadConfig, run_load
from .sources import SourceExplorer
from .task import ELTTask
from .utils import format_df, truncate
from .warehouse import Destination, Warehouse
from .workspace import Workspace

MAX_ROWS = 50


def _tool(fn):
    """Count calls/errors (for metrics & process signals), turn exceptions into feedback."""
    @functools.wraps(fn)
    async def wrapper(self: ELTSession, *args, **kwargs) -> str:
        self.tool_calls[fn.__name__] += 1
        try:
            out = await fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - every failure becomes model feedback
            self.tool_errors[fn.__name__] += 1
            out = f"ERROR: {type(e).__name__}: {e}"
        return truncate(out, self.max_output_chars)
    return wrapper


class ELTSession:

    def __init__(self, task: ELTTask, workspace: Workspace, warehouse: Warehouse,
                 profiles_dir: Path, max_output_chars: int = 4000):
        self.task = task
        self.workspace = workspace
        self.warehouse = warehouse
        self.profiles_dir = profiles_dir
        self.max_output_chars = max_output_chars
        self.submitted = False
        self.tool_calls: Counter[str] = Counter()
        self.tool_errors: Counter[str] = Counter()
        self._explorer: SourceExplorer | None = None

    @classmethod
    def create(cls, task: ELTTask, destination: Destination,
               base_dir: str | Path | None = None) -> ELTSession:
        workspace = Workspace.create(task, base_dir)
        warehouse = destination.create_warehouse(namespace=workspace.root.name)
        profiles_dir = Path(tempfile.mkdtemp(prefix="profiles_", dir=base_dir))
        write_profiles(profiles_dir, warehouse)
        return cls(task, workspace, warehouse, profiles_dir)

    @_tool
    async def list_files(self, path: str = ".") -> str:
        """List files in the workspace (task spec, source data, your dbt models)."""
        return "\n".join(self.workspace.list_files(path)) or "(empty)"

    @_tool
    async def read_file(self, path: str) -> str:
        """Read a text file from the workspace."""
        return self.workspace.resolve(path).read_text(encoding="utf-8", errors="replace")

    @_tool
    async def write_file(self, path: str, content: str) -> str:
        """Create/overwrite load_config.yaml or a dbt file under models/."""
        p = self.workspace.resolve_writable(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} chars)"

    @_tool
    async def query_source(self, sql: str) -> str:
        """Read-only SQL over source data; tables are named <source>.<table>, e.g. crm.customers."""
        if self._explorer is None:
            self._explorer = await asyncio.to_thread(SourceExplorer, self.task.source_tables)
        return format_df(*await asyncio.to_thread(self._explorer.query, sql, MAX_ROWS))

    @_tool
    async def run_load(self) -> str:
        """Extract & load according to load_config.yaml."""
        cfg_path = self.workspace.resolve(LOAD_CONFIG_FILE)
        if not cfg_path.exists():
            return f"ERROR: write {LOAD_CONFIG_FILE} first"
        config = LoadConfig.parse(cfg_path.read_text(encoding="utf-8"))
        return await asyncio.to_thread(run_load, self.task, config, self.warehouse)

    @_tool
    async def run_dbt(self) -> str:
        """Run `dbt run` on the project; returns the log (errors included)."""
        ok, log = await run_dbt_project(self.workspace.root, self.profiles_dir)
        return ("dbt run succeeded.\n" if ok else "dbt run FAILED.\n") + log

    @_tool
    async def query_warehouse(self, sql: str) -> str:
        """Read-only SQL on the warehouse (raw tables and your models)."""
        return format_df(*await asyncio.to_thread(self.warehouse.query, sql, MAX_ROWS))

    @_tool
    async def submit(self) -> str:
        """Finish the episode; the pipeline is then graded."""
        self.submitted = True
        return "Submitted. Your pipeline will now be graded."

    async def close(self) -> None:
        if self._explorer is not None:
            self._explorer.close()
        self.warehouse.drop()
        self.workspace.cleanup()
        shutil.rmtree(self.profiles_dir, ignore_errors=True)
