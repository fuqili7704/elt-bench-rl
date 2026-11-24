"""Generate ground_truth/<model>.csv for a local task by running its reference solution.

Usage:  uv run python scripts/make_ground_truth.py tasks/toy_shop
"""

import asyncio
import sys

from elt_env.grader import build_pipeline, reference_submission
from elt_env.task import ELTTask
from elt_env.warehouse import DuckDBDestination


async def main(task_dir: str) -> None:
    task = ELTTask.load(task_dir)
    session, dbt_ok = await build_pipeline(task, reference_submission(task), DuckDBDestination())
    try:
        if not dbt_ok:
            sys.exit("reference solution failed to run")
        out = task.task_dir / "ground_truth"
        out.mkdir(exist_ok=True)
        for model in task.model_names:
            df = session.warehouse.fetch_table(session.warehouse.model_schema, model)
            df.to_csv(out / f"{model}.csv", index=False)
            print(f"--- {model} ({len(df)} rows) ---\n{df.to_string(index=False)}\n")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
