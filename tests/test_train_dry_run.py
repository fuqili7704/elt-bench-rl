"""One GRPO step with scripted policies on DuckDB (no Tinker). Downloads the tokenizer once."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402


async def test_dry_run_group_on_toy_shop():
    args = train.parse_args(["--dry-run", "--task", "tasks/toy_shop", "--destination", "duckdb",
                             "--group-size", "3", "--max-turns", "15"])
    try:
        out = await train.dry_run(args, ROOT / "tasks" / "toy_shop")
    except OSError as e:  # tokenizer download unavailable
        pytest.skip(f"tokenizer not available: {e}")

    rewards = {r["policy"]: r["reward"] for r in out["rows"]}
    assert rewards["solve (reference SQL)"] == pytest.approx(1.0)
    assert rewards["load only, no models"] == pytest.approx(0.2)
    assert rewards["give up immediately"] == pytest.approx(0.0)

    advantages = [r["advantage"] for r in out["rows"]]
    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    assert advantages[0] > 0 > advantages[2]
    assert out["num_datums"] >= 3 and all(n > 0 for n in out["datum_tokens"])
