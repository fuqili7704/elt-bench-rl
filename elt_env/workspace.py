"""Per-rollout sandbox directory (the "exam room").

Only agent-visible task files are copied in, and every path the agent passes
is checked to stay inside the room.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .dbt_runner import write_dbt_project
from .loader import LOAD_CONFIG_FILE
from .task import AGENT_VISIBLE, ELTTask

WRITABLE_MODEL_SUFFIXES = {".sql", ".yml", ".yaml"}
HIDDEN_DIRS = {"target", "logs", "dbt_packages"}  # dbt build artifacts, noise for the agent


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, task: ELTTask, base_dir: str | Path | None = None) -> Workspace:
        root = Path(tempfile.mkdtemp(prefix=f"ws_{task.task_id}_", dir=base_dir))
        for name in AGENT_VISIBLE:
            src = task.task_dir / name
            if src.is_dir():
                shutil.copytree(src, root / name)
            elif src.is_file():
                shutil.copy2(src, root / name)
        write_dbt_project(root)
        return cls(root)

    def resolve(self, rel_path: str) -> Path:
        """Map an agent-supplied path into the workspace; refuse anything outside it.

        Covers '../..', absolute paths and symlinks (resolve() follows them).
        """
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"path {rel_path!r} is outside the workspace")
        return p

    def resolve_writable(self, rel_path: str) -> Path:
        p = self.resolve(rel_path)
        rel = p.relative_to(self.root)
        if rel.as_posix() == LOAD_CONFIG_FILE:
            return p
        if rel.parts and rel.parts[0] == "models" and p.suffix in WRITABLE_MODEL_SUFFIXES:
            return p
        raise PermissionError(
            f"cannot write {rel_path!r}: only '{LOAD_CONFIG_FILE}' and models/**/*.sql|.yml are writable"
        )

    def list_files(self, rel_path: str = ".", max_entries: int = 200) -> list[str]:
        base = self.resolve(rel_path)
        out = []
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(self.root)
            if p.is_file() and not (set(rel.parts) & HIDDEN_DIRS):
                out.append(rel.as_posix())
                if len(out) >= max_entries:
                    out.append("... (more files omitted)")
                    break
        return out

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
