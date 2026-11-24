"""Run the agent's dbt project in a subprocess (dbt isn't safe to run concurrently in-process)."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import yaml

from .warehouse import Warehouse

PROFILE_NAME = "elt"

# Owned by the environment; the agent may not overwrite it (no hooks, no
# python models, no custom paths).
DBT_PROJECT_YML = f"""\
name: elt_pipeline
version: "1.0.0"
config-version: 2
profile: {PROFILE_NAME}
model-paths: ["models"]
models:
  elt_pipeline:
    +materialized: table
"""


def write_dbt_project(project_dir: Path) -> None:
    (project_dir / "models").mkdir(parents=True, exist_ok=True)
    (project_dir / "dbt_project.yml").write_text(DBT_PROJECT_YML, encoding="utf-8")


def write_profiles(profiles_dir: Path, warehouse: Warehouse) -> None:
    """Kept outside the workspace since it can hold credentials."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile = {PROFILE_NAME: {"target": "dev", "outputs": {"dev": warehouse.dbt_profile()}}}
    (profiles_dir / "profiles.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")


def _dbt_executable() -> str:
    exe = Path(sys.executable).parent / ("dbt.exe" if os.name == "nt" else "dbt")
    return str(exe) if exe.exists() else (shutil.which("dbt") or "dbt")


def _clean_env() -> dict[str, str]:
    """Minimal env: secrets such as TINKER_API_KEY must not be readable via {{ env_var() }}."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    env["DBT_SEND_ANONYMOUS_USAGE_STATS"] = "False"
    return env


async def run_dbt_project(project_dir: Path, profiles_dir: Path, timeout_s: float = 180) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        _dbt_executable(), "run",
        "--project-dir", str(project_dir), "--profiles-dir", str(profiles_dir), "--no-use-colors",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        cwd=str(project_dir), env=_clean_env(),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"dbt run timed out after {timeout_s:.0f}s"
    return proc.returncode == 0, out.decode(errors="replace")
