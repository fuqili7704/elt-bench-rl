"""Full episodes through tinker-cookbook's env with a scripted policy, no credentials needed."""

import json
from pathlib import Path

import pytest
from tinker_cookbook.renderers.base import ToolCall

from elt_env.grader import reference_submission
from elt_env.rl_env import build_message_env, system_prompt, task_prompt
from elt_env.task import ELTTask
from elt_env.tools import ELTSession
from elt_env.warehouse import DuckDBDestination

TASK = ELTTask.load(Path(__file__).resolve().parents[1] / "tasks" / "toy_shop")
REF = reference_submission(TASK)


def call(name: str, **arguments) -> dict:
    tc = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)), id=f"call_{name}")
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


@pytest.fixture
async def env_and_session(tmp_path):
    dest = DuckDBDestination(tmp_path / "wh")
    session = ELTSession.create(TASK, dest, tmp_path)
    initial = [{"role": "system", "content": system_prompt(session)},
               {"role": "user", "content": task_prompt(TASK)}]
    env = build_message_env(session, dest, initial, max_turns=20)
    yield env, session
    await session.close()


async def run_policy(env, messages):
    await env.initial_observation()
    results = []
    for m in messages:
        r = await env.step(m)
        results.append(r)
        if r.episode_done:
            break
    return results


async def test_scripted_solution_gets_full_reward(env_and_session):
    env, session = env_and_session
    policy = [
        call("list_files"),
        call("query_source", sql="SELECT * FROM orders_store.orders LIMIT 3"),
        call("write_file", path="load_config.yaml", content=REF["load_config.yaml"]),
        call("run_load"),
        *[call("write_file", path=p, content=c) for p, c in REF.items() if p.startswith("models/")],
        call("run_dbt"),
        call("query_warehouse", sql="SELECT * FROM analytics.city_sales"),
        call("submit"),
    ]
    results = await run_policy(env, policy)

    last = results[-1]
    assert last.episode_done and len(results) == len(policy)
    assert last.reward == pytest.approx(1.0), last.metrics
    assert last.metrics["submitted"] == 1.0
    assert all(r.reward == 0.0 for r in results[:-1])
    assert any("loaded raw.orders (10 rows)" in m.get("content", "") for m in env.history if m["role"] == "tool")


async def test_stopping_without_tools_ends_and_grades_zero(env_and_session):
    env, _ = env_and_session
    results = await run_policy(env, [{"role": "assistant", "content": "I give up."}])
    assert results[0].episode_done and results[0].reward == 0.0


async def test_max_turns_ends_episode(tmp_path):
    dest = DuckDBDestination(tmp_path / "wh")
    session = ELTSession.create(TASK, dest, tmp_path)
    try:
        env = build_message_env(session, dest, [{"role": "user", "content": "go"}], max_turns=3)
        results = await run_policy(env, [call("list_files")] * 10)
        assert len(results) == 3 and results[-1].episode_done
    finally:
        await session.close()


async def test_load_only_policy_gets_partial_reward(env_and_session):
    env, _ = env_and_session
    results = await run_policy(env, [
        call("write_file", path="load_config.yaml", content=REF["load_config.yaml"]),
        call("run_load"),
        call("submit"),
    ])
    assert results[-1].reward == pytest.approx(0.2)


async def test_cheating_policy_gets_nothing(env_and_session):
    env, session = env_and_session
    secret = (TASK.task_dir / "reference" / "models" / "city_sales.sql").as_posix()
    results = await run_policy(env, [
        call("read_file", path="../reference/models/city_sales.sql"),
        call("query_source", sql=f"SELECT * FROM read_text('{secret}')"),
        call("write_file", path="models/cheat.py", content="def model(dbt, session): ..."),
        call("submit"),
    ])
    tool_outputs = [m["content"] for m in env.history if m["role"] == "tool"]
    assert all(o.startswith("ERROR") for o in tool_outputs[:3]), tool_outputs
    assert results[-1].reward == 0.0
    assert results[-1].metrics["num_tool_errors"] == 3.0


async def test_malformed_tool_call_is_penalized_not_fatal(env_and_session):
    env, _ = env_and_session
    await env.initial_observation()
    bad = {"role": "assistant", "content": "", "tool_calls": [
        ToolCall(function=ToolCall.FunctionBody(name="run_load", arguments="{not json"), id="x")]}
    r = await env.step(bad)
    assert not r.episode_done
    assert "error" in env.history[-1]["content"].lower()
