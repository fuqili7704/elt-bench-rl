"""GRPO training on ELT-Bench using tinker-cookbook's RL loop.

    uv run python scripts/train.py                       # 1 step on address (Snowflake)
    uv run python scripts/train.py --dry-run             # scripted policies, no Tinker calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import Message, RenderContext, ToolCall
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import TrajectoryGroup

from elt_env.grader import reference_submission
from elt_env.rl_env import ELTDatasetBuilder, ELTEnvGroupBuilder
from elt_env.task import ELTTask
from elt_env.warehouse.snowflake_wh import load_env_file

ROOT = Path(__file__).resolve().parents[1]


async def train(args, task_dir: Path) -> None:
    from tinker_cookbook.rl import train as rl_train

    log_path = ROOT / "runs" / f"train_{task_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    config = rl_train.Config(
        model_name=args.model,
        recipe_name="elt_bench_rl",
        dataset_builder=ELTDatasetBuilder(
            task_dirs=[str(task_dir)],
            model_name=args.model,
            batch_size=1,
            group_size=args.group_size,
            destination=args.destination,
            max_turns=args.max_turns,
            max_trajectory_tokens=args.context_window,
        ),
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        max_steps=args.steps,
        num_epochs=args.steps,
        eval_every=0,
        save_every=0,
        log_path=str(log_path),
    )
    await rl_train.main(config)

    metrics_file = log_path / "metrics.jsonl"
    if metrics_file.is_file():
        print("\n" + "=" * 70)
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            keys = [k for k in sorted(row) if any(s in k for s in
                    ("reward", "load_score", "model_score", "official_pass", "submitted", "loss", "step"))]
            print({k: row[k] for k in keys})
    print(f"\nlogs in {log_path.relative_to(ROOT)}")


def _call(name: str, **arguments) -> Message:
    tc = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)), id=f"call_{name}")
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


def _load_config(task: ELTTask) -> str:
    ref = task.reference_dir / "load_config.yaml"
    if ref.is_file():
        return ref.read_text(encoding="utf-8")
    streams = [{"source": t.source, "table": t.table, "destination_table": t.table} for t in task.source_tables]
    return yaml.safe_dump({"destination_schema": task.raw_schema, "streams": streams}, sort_keys=False)


def scripted_policies(task: ELTTask) -> dict[str, list[Message]]:
    """Behaviours with different expected rewards, so the group has nonzero advantages."""
    load = [_call("list_files"),
            _call("write_file", path="load_config.yaml", content=_load_config(task)),
            _call("run_load")]
    scripts: dict[str, list[Message]] = {}
    if (task.reference_dir / "models").is_dir():
        models = [_call("write_file", path=p, content=c)
                  for p, c in reference_submission(task).items() if p.startswith("models/")]
        scripts["solve (reference SQL)"] = [*load, *models, _call("run_dbt"), _call("submit")]
    scripts["load only, no models"] = [*load, _call("submit")]
    scripts["give up immediately"] = [{"role": "assistant", "content": "I cannot do this."}]
    return scripts


class ScriptedPolicy(TokenCompleter):
    """A TokenCompleter that 'samples' a fixed list of assistant messages, rendered to tokens."""

    def __init__(self, renderer, messages: list[Message]):
        self.renderer = renderer
        self.messages = messages
        self.i = 0

    async def __call__(self, model_input, stop, *, max_tokens=None) -> TokensWithLogprobs:
        if self.i < len(self.messages):
            msg = self.messages[self.i]
        else:
            msg = {"role": "assistant", "content": "Done."}
        self.i += 1
        rendered = self.renderer.render_message(msg, RenderContext(idx=1, is_last=True))
        tokens = [tok for chunk in rendered.output for tok in chunk.tokens]
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[0.0] * len(tokens))


async def dry_run(args, task_dir: Path) -> dict:
    task = ELTTask.load(task_dir)
    scripts = list(scripted_policies(task).items())
    names = [scripts[i % len(scripts)][0] for i in range(args.group_size)]
    by_name = dict(scripts)

    builder = ELTEnvGroupBuilder(str(task_dir), args.model, group_size=args.group_size,
                                 destination=args.destination, max_turns=args.max_turns,
                                 max_trajectory_tokens=args.context_window)
    t0 = time.time()
    envs = await builder.make_envs()
    try:
        trajs = await asyncio.gather(*[do_single_rollout(ScriptedPolicy(env.renderer, by_name[n]), env)
                                       for env, n in zip(envs, names)])
        final = await builder.compute_group_rewards(list(trajs), envs)
    finally:
        await builder.cleanup()
    elapsed = time.time() - t0

    group = TrajectoryGroup(list(trajs), [r for r, _ in final], [m for _, m in final])
    advantages = compute_advantages([group])[0].tolist()
    data, _ = assemble_training_data([group], [compute_advantages([group])[0]])

    print("=" * 70)
    print(f"DRY RUN  task {task_dir.name} | destination {args.destination} | group_size {args.group_size} "
          f"| {elapsed:.0f}s")
    rows = []
    for name, traj, reward, adv in zip(names, trajs, group.get_total_rewards(), advantages):
        m = traj.transitions[-1].metrics
        row = {"policy": name, "turns": len(traj.transitions), "reward": reward, "advantage": adv,
               **{k: m[k] for k in ("load_score", "model_score", "official_pass", "submitted") if k in m}}
        rows.append(row)
        print(f"  {name:24s} turns {row['turns']:2d}  reward {reward:+.2f}  advantage {adv:+.2f}  "
              + "  ".join(f"{k} {row[k]:.2f}" for k in ("load_score", "model_score", "official_pass") if k in row))
    n_tokens = [d.model_input.length for d in data]
    print(f"training datums {len(data)} (tokens per datum {n_tokens}); "
          "a real step sends these to forward_backward, then optim_step.")
    return {"rows": rows, "num_datums": len(data), "datum_tokens": n_tokens}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="tasks/elt_bench/address")
    ap.add_argument("--destination", default="snowflake", choices=["duckdb", "snowflake"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=2048, help="max new tokens per model turn")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="scripted policies, no Tinker calls")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    load_env_file(ROOT / ".env")
    task_dir = (ROOT / args.task).resolve()
    asyncio.run(dry_run(args, task_dir) if args.dry_run else train(args, task_dir))


if __name__ == "__main__":
    main()
