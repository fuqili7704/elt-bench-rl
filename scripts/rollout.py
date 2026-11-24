"""Run one episode with a Tinker model and print the transcript and grade.

    uv run python scripts/rollout.py
    uv run python scripts/rollout.py --task tasks/toy_shop --destination duckdb
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_single_rollout

from elt_env.rl_env import ELTEnvGroupBuilder
from elt_env.warehouse.snowflake_wh import load_env_file

ROOT = Path(__file__).resolve().parents[1]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def print_transcript(history: list[dict], max_chars: int) -> None:
    for i, m in enumerate(history):
        role = m["role"]
        if role == "system":
            continue
        body = _text(m.get("content", ""))
        for tc in m.get("tool_calls") or []:
            body += f"\n  -> {tc.function.name}({tc.function.arguments})"
        if len(body) > max_chars:
            body = body[:max_chars] + f" ... [{len(body) - max_chars} more chars]"
        print(f"\n[{i}] {role.upper()}: {body}")


def to_jsonable(history: list[dict]) -> list[dict]:
    out = []
    for m in history:
        d = {"role": m["role"], "content": _text(m.get("content", ""))}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"name": tc.function.name, "arguments": tc.function.arguments}
                               for tc in m["tool_calls"]]
        out.append(d)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="tasks/elt_bench/address")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--destination", default="snowflake", choices=["duckdb", "snowflake"])
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048, help="max new tokens per model turn")
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--show-chars", type=int, default=600, help="truncate each printed message")
    args = ap.parse_args()

    load_env_file(ROOT / ".env")
    task_dir = (ROOT / args.task).resolve()
    builder = ELTEnvGroupBuilder(str(task_dir), args.model, group_size=1, destination=args.destination,
                                 max_turns=args.max_turns, max_trajectory_tokens=args.context_window)

    service = tinker.ServiceClient()
    sampling = await service.create_sampling_client_async(base_model=args.model)
    policy = TinkerTokenCompleter(sampling, max_tokens=args.max_tokens, temperature=args.temperature,
                                  context_window=args.context_window)

    env = (await builder.make_envs())[0]
    msg_env = env.message_env
    t0 = time.time()
    try:
        traj = await do_single_rollout(policy, env)
    finally:
        elapsed = time.time() - t0
        await builder.cleanup()

    last = traj.transitions[-1]
    grade = msg_env.reward_fn.result
    sampled = sum(len(t.ac.tokens) for t in traj.transitions)
    prompt = sum(t.ob.length for t in traj.transitions)

    print_transcript(msg_env.history, args.show_chars)
    print("\n" + "=" * 70)
    print(f"task {task_dir.name} | model {args.model} | destination {args.destination}")
    print(f"turns {len(traj.transitions)} | stop {traj.stop_reason} | {elapsed:.0f}s | "
          f"tokens sampled {sampled}, prompt tokens {prompt}")
    print(f"episode reward {sum(t.reward for t in traj.transitions):.3f} (final-step reward {last.reward:.3f})")
    for k, v in sorted(last.metrics.items()):
        if isinstance(v, (int, float)):
            print(f"  {k:28s} {v:.3f}")
    if grade is not None:
        for table, why in grade.details.items():
            print(f"  why {table}: {why}")

    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"rollout_{task_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "task": task_dir.name, "model": args.model, "destination": args.destination,
        "elapsed_s": elapsed, "tokens_sampled": sampled, "prompt_tokens": prompt,
        "episode_reward": sum(t.reward for t in traj.transitions),
        "metrics": {k: v for k, v in last.metrics.items() if isinstance(v, (int, float))},
        "grade_details": grade.details if grade else None,
        "transcript": to_jsonable(msg_env.history),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
