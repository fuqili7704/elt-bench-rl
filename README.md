# elt-bench-rl

An RL environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench), packaged as a
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipe.

In each episode the model gets an ELT-Bench task and a set of tools. It has to:

1. load the source tables into a warehouse;
2. write dbt models for the requested tables;
3. submit.

On submit, the pipeline is rebuilt from scratch in a clean warehouse and compared with the
official ground truth. The result of that comparison is the reward. Training uses the cookbook's
RL loop; this repo provides the environment.

Supported warehouses:

- **DuckDB**: local and free, used by the tests.
- **Snowflake**: the warehouse ELT-Bench is graded on.

To add another warehouse, implement one class in `elt_env/warehouse/`.

## Quick start

Requirements: Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Tested on Ubuntu (WSL).

```bash
uv sync --extra dev
uv run pytest -q
```

This runs every test that doesn't need credentials (see "Local tests" below).

The Snowflake and Tinker steps need credentials. Copy `.env.example` to `.env` and fill in the
parts you need. `.env` is git-ignored.

## Repository layout

```
elt_env/
  task.py, sources.py      task spec and source readers
  loader.py                the EL step: copies source tables into the warehouse
  workspace.py, tools.py   the agent's sandbox and its 8 tools
  dbt_runner.py            runs dbt
  grader.py                rebuilds a submission and scores it
  official_compare.py      ELT-Bench's comparison rules (stage 1 and stage 2)
  warehouse/               DuckDB and Snowflake backends
  rl_env.py                glue to tinker-cookbook (env, reward, dataset builder)
tasks/
  toy_shop/                small task with a reference solution, used by the tests
  elt_bench/address/       an official ELT-Bench task
scripts/
  import_elt_bench.py      converts an official task into this layout
  rollout.py               one episode with a Tinker model
  train.py                 RL training steps (has a --dry-run mode)
  check_snowflake.py       quick Snowflake connection check
  make_ground_truth.py     regenerates the toy task's expected tables from its reference solution
tests/
```

## Episodes and scoring

Tools available to the agent:

- `list_files`, `read_file`, `write_file`
- `query_source`
- `run_load`, `run_dbt`
- `query_warehouse`
- `submit`

An episode ends when the agent does any of the following:

- calls `submit`;
- replies without a tool call;
- reaches the turn limit or the token limit.

Grading uses only two outputs of the episode: `load_config.yaml` and `models/`. Both are re-run
in a fresh warehouse, so changes the agent made to its own warehouse during the episode have no
effect on the score.

```
reward = 0.2 * load_score + 0.8 * model_score
```

- **load_score**: the fraction of raw tables that match the source data. This is ELT-Bench's
  stage 1 rule (`eva_stage1_stored.py`).
- **model_score**: the fraction of data models that match the ground truth. This is the stage 2
  rule (`eva_stage2.py`):
  - columns are matched by name;
  - rows are sorted by key;
  - numbers are compared with a small tolerance.

`official_pass` is logged as well. It follows the benchmark's own success criterion: the model
score counts only if every raw table loaded correctly.

The agent cannot:

- read anything outside its workspace;
- write files other than `load_config.yaml` and dbt `.sql`/`.yml` files;
- run SQL that isn't read-only;
- see the ground truth or the reference solution.

dbt also runs with a cleaned-up environment, so API keys are not visible to model SQL. These
restrictions are tested in `tests/test_rl_env.py` and `tests/test_tools.py`.

## Local tests (no credentials)

```bash
uv run pytest -q
```

All tests run on DuckDB against `tasks/toy_shop`. Two of them cover the whole loop.

**`test_rl_env.py`** plays scripted episodes through the cookbook's `AgentToolMessageEnv`:

- a correct solution gets 1.0;
- loading only gets 0.2;
- giving up gets 0;
- cheating attempts are blocked and score 0.

**`test_train_dry_run.py`** runs one GRPO step without Tinker:

1. play a group of scripted episodes through the token-level env;
2. grade them;
3. build the training data with the cookbook's own `compute_advantages` and
   `assemble_training_data`.

It downloads the Qwen tokenizer from Hugging Face on the first run.

The Snowflake tests are skipped unless `.env` is set up.

## Running on an official task (Snowflake + Tinker)

### 1. Get the task data

Only the data files from ELT-Bench are needed. Its Docker setup is not, because the loader reads
the dumps directly.

The three downloads are the same ones `setup/elt_setup.sh` makes in the ELT-Bench repo:

```bash
git clone https://github.com/uiuc-kang-lab/ELT-Bench.git
cd ELT-Bench/setup
pip install gdown
gdown 'https://drive.google.com/uc?id=1qVAzU3kgn_G72QQ4b5zt3e1hwkQcSgDq'
gdown 'https://drive.google.com/uc?id=1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0'
gdown 'https://drive.google.com/uc?id=11vQqNEWXoPG6sjKytAn7TtFLDMiQa17I'
unzip data_api.zip -d ../elt-docker/rest_api
unzip data_db.zip -d ./
cd ..
pip install -U huggingface_hub
hf download tttjjj/elt_bench --repo-type dataset --include "gt_snowflake/**" --local-dir ./ground_truth
```

Then, from this repo:

```bash
uv run python scripts/import_elt_bench.py address --elt-bench /path/to/ELT-Bench
```

This fills in `tasks/elt_bench/address/source/` and `ground_truth/`. Both are git-ignored.

### 2. Set up Snowflake

Use a key-pair service user with its own role and warehouse.

First generate a key pair. Keep the private key on your machine and never commit it:

```bash
mkdir -p ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/elt_rl_key.p8 -nocrypt
openssl rsa -in ~/.snowflake/elt_rl_key.p8 -pubout -out ~/.snowflake/elt_rl_key.pub
chmod 600 ~/.snowflake/elt_rl_key.p8
```

Then run the following as ACCOUNTADMIN:

```sql
CREATE ROLE ELT_RL_ROLE;
CREATE WAREHOUSE ELT_RL_WH WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 INITIALLY_SUSPENDED = TRUE;
GRANT USAGE ON WAREHOUSE ELT_RL_WH TO ROLE ELT_RL_ROLE;
GRANT CREATE DATABASE ON ACCOUNT TO ROLE ELT_RL_ROLE;
CREATE USER ELT_RL_USER TYPE = SERVICE DEFAULT_ROLE = ELT_RL_ROLE DEFAULT_WAREHOUSE = ELT_RL_WH
  RSA_PUBLIC_KEY = '<public key, without the BEGIN/END lines>';
GRANT ROLE ELT_RL_ROLE TO USER ELT_RL_USER;
```

Putting the warehouse under a resource monitor is recommended. It keeps a runaway run from using
up the credits.

Fill in the `SNOWFLAKE_*` values in `.env`, then check the connection:

```bash
uv run python scripts/check_snowflake.py
uv run pytest -q tests/test_snowflake.py
```

If `address` has been imported, the second command also checks that the official answer for it
scores 1.0 on Snowflake.

### 3. Run one rollout

Add `TINKER_API_KEY` to `.env`, then:

```bash
uv run python scripts/rollout.py
```

By default this runs `address` on Snowflake with `Qwen/Qwen3.5-4B`. To use a different model that
your Tinker account supports, pass `--model`.

The script prints the conversation and the score, and saves a copy to `runs/`. Each episode gets
its own Snowflake database, which is dropped at the end.

### 4. Run one training step

```bash
uv run python scripts/train.py
```

This calls `tinker_cookbook.rl.train.main` with `ELTDatasetBuilder`. By default it runs one step
on `address`:

1. sample 4 episodes;
2. grade each one by re-running it;
3. compute group-relative advantages;
4. run one LoRA update on Tinker.

Logs are written to `runs/train_*/`.

Cheaper variants:

```bash
uv run python scripts/train.py --task tasks/toy_shop --destination duckdb   # local warehouse
uv run python scripts/train.py --dry-run                                    # no Tinker at all
```

## Limitations

- ELT-Bench uses Airbyte for loading. This repo uses a lightweight in-process loader instead,
  and checks its output with the official stage 1 rule. An Airbyte backend for comparison is
  planned.
- ELT-Bench does not publish reference SQL for its tasks, so `address` is graded against the
  ground-truth tables only.
