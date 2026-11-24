"""Tinker RL integration: plug ELTSession into tinker-cookbook's tool-use env."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from .grader import GradeResult, collect_submission, grade
from .task import ELTTask
from .tools import ELTSession
from .warehouse import Destination, DuckDBDestination, SnowflakeDestination

logger = logging.getLogger(__name__)

FAILED_PARSE_REWARD = -0.1


def make_destination(name: str) -> Destination:
    if name == "duckdb":
        return DuckDBDestination()
    if name == "snowflake":
        return SnowflakeDestination.from_env()
    raise ValueError(f"unknown destination {name!r}")


def system_prompt(session: ELTSession) -> str:
    wh = session.warehouse
    raw = session.task.raw_schema
    return f"""\
You are a data engineer building an ELT pipeline with tools.

Workspace layout:
- task.yaml: the task, its sources, and the schema where raw tables must land ("{raw}")
- data_model.yaml: the target tables (data models) you must build, with column definitions
- source/: source data files (you can also query them with query_source)
- load_config.yaml: YOU write this to configure extraction & loading
- models/: YOU write dbt models here, one <model_name>.sql file per data model

load_config.yaml format:
destination_schema: {raw}
streams:
  - {{source: <source name>, table: <table name>, destination_table: <table name>}}

dbt notes:
- Declare the loaded raw tables in models/sources.yml (source name "{raw}", schema "{raw}"),
  then select from them with {{{{ source('{raw}', '<table>') }}}}.
- Reference another model with {{{{ ref('<model_name>') }}}}.
- Every model is materialized as a table in schema "{wh.model_schema}". SQL dialect: {wh.name}.

Workflow: inspect the task -> write load_config.yaml -> run_load -> write models -> run_dbt
-> fix errors -> verify results with query_warehouse -> submit.
Grading re-runs ONLY load_config.yaml and models/ from scratch in a fresh warehouse and compares
each data model with a hidden expected table (column names and values must match exactly).
"""


def task_prompt(task: ELTTask) -> str:
    data_model = (task.task_dir / "data_model.yaml").read_text(encoding="utf-8")
    return (
        f"{task.description}\n"
        f"data_model.yaml:\n```yaml\n{data_model}```\n"
        "Begin by inspecting the workspace."
    )


class ELTTools:
    """Wrap ELTSession methods as tinker-cookbook tools."""

    def __init__(self, session: ELTSession):
        self.session = session

    @tool
    async def list_files(
        self, path: Annotated[str, "Directory relative to the workspace root."] = "."
    ) -> ToolResult:
        """List all files in the workspace (task spec, source data, your dbt models)."""
        return simple_tool_result(await self.session.list_files(path))

    @tool
    async def read_file(self, path: Annotated[str, "File path relative to the workspace root."]) -> ToolResult:
        """Read a text file from the workspace."""
        return simple_tool_result(await self.session.read_file(path))

    @tool
    async def write_file(
        self,
        path: Annotated[str, "Either 'load_config.yaml' or a path under models/ ending in .sql or .yml."],
        content: Annotated[str, "Full new content of the file."],
    ) -> ToolResult:
        """Create or overwrite load_config.yaml or a dbt file under models/."""
        return simple_tool_result(await self.session.write_file(path, content))

    @tool
    async def query_source(
        self, sql: Annotated[str, "A SQL query. Source tables are named <source>.<table>, e.g. crm.customers."]
    ) -> ToolResult:
        """Run a read-only SQL query over the SOURCE data (before loading). Returns at most 50 rows."""
        return simple_tool_result(await self.session.query_source(sql))

    @tool
    async def run_load(self) -> ToolResult:
        """Extract and load source tables into the warehouse according to load_config.yaml."""
        return simple_tool_result(await self.session.run_load())

    @tool
    async def run_dbt(self) -> ToolResult:
        """Run `dbt run` on your models and return the log, including any errors."""
        return simple_tool_result(await self.session.run_dbt())

    @tool
    async def query_warehouse(
        self, sql: Annotated[str, "A read-only SQL query on the warehouse (raw tables and your models)."]
    ) -> ToolResult:
        """Run a read-only SQL query on the WAREHOUSE to check loaded tables and model results. Returns at most 50 rows."""
        return simple_tool_result(await self.session.query_warehouse(sql))

    @tool
    async def submit(self) -> ToolResult:
        """Submit your pipeline and end the episode. Call this when you are done."""
        return simple_tool_result(await self.session.submit(), should_stop=True)

    def all(self) -> list:
        return [self.list_files, self.read_file, self.write_file, self.query_source,
                self.run_load, self.run_dbt, self.query_warehouse, self.submit]


@dataclass
class ELTReward:
    """Episode-end reward: grade the submitted pipeline by re-running it."""

    session: ELTSession
    destination: Destination
    result: GradeResult | None = None

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.result = await grade(self.session.task, collect_submission(self.session.workspace), self.destination)
        metrics = self.result.metrics()
        metrics["submitted"] = float(self.session.submitted)
        metrics["num_tool_calls"] = float(sum(self.session.tool_calls.values()))
        metrics["num_tool_errors"] = float(sum(self.session.tool_errors.values()))
        return self.result.reward, metrics


def build_message_env(session: ELTSession, destination: Destination,
                      initial_messages: list[Message], max_turns: int) -> AgentToolMessageEnv:
    tools = ELTTools(session)
    return AgentToolMessageEnv(
        tools=tools.all(),
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=ELTReward(session, destination),
        failed_parse_reward=FAILED_PARSE_REWARD,
        terminate_on_parse_error=False,
        tool_execution="sequential",
    )


class ELTEnvGroupBuilder(EnvGroupBuilder):
    """group_size independent rollouts of ONE task (GRPO compares within the group)."""

    def __init__(self, task_dir: str, model_name: str, group_size: int,
                 renderer_name: str | None = None, destination: str = "duckdb",
                 max_turns: int = 20, max_trajectory_tokens: int = 32 * 1024):
        self.task_dir = task_dir
        self.model_name = model_name
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.destination = destination
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self._sessions: list[ELTSession] = []

    async def make_envs(self) -> Sequence[Env]:
        task = ELTTask.load(self.task_dir)
        destination = make_destination(self.destination)
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        envs: list[Env] = []
        for _ in range(self.group_size):
            session = ELTSession.create(task, destination)
            self._sessions.append(session)
            specs = [t.to_spec() for t in ELTTools(session).all()]
            prefix = renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=system_prompt(session))
            initial = prefix + [{"role": "user", "content": task_prompt(task)}]
            msg_env = build_message_env(session, destination, initial, self.max_turns)
            envs.append(EnvFromMessageEnv(
                renderer=renderer,
                message_env=msg_env,
                failed_parse_reward=FAILED_PARSE_REWARD,
                terminate_on_parse_error=False,
                max_trajectory_tokens=self.max_trajectory_tokens,
            ))
        return envs

    async def cleanup(self) -> None:
        for s in self._sessions:
            try:
                await s.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("session cleanup failed: %s", e)
        self._sessions.clear()

    def logging_tags(self) -> list[str]:
        return ["elt", f"elt_{self.destination}"]


class ELTDataset(RLDataset):

    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size:(index + 1) * self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):

    task_dirs: list[str]
    model_name: str
    batch_size: int = 1
    group_size: int = 4
    renderer_name: str | None = None
    destination: str = "duckdb"
    max_turns: int = 20
    max_trajectory_tokens: int = 32 * 1024

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        builders = [
            ELTEnvGroupBuilder(d, self.model_name, self.group_size, self.renderer_name,
                               self.destination, self.max_turns, self.max_trajectory_tokens)
            for d in self.task_dirs
        ]
        return ELTDataset(builders, self.batch_size), None
