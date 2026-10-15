# elt-rlvr: ELT-Bench as a Tinker RLVR environment

A reinforcement-learning-with-verifiable-rewards recipe for
[ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) built on
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
In each rollout a language model reads the task specification, configures and
runs extraction and loading, writes dbt transformations, uses execution feedback,
and submits. The environment grades the resulting warehouse state with
ELT-Bench's own evaluators and returns the score as the reward.

## The three checks

| Check | Command | Needs |
|---|---|---|
| 1. Credential-free local integration test | `pytest` (or `elt-rlvr demo-local`) | Python only. No accounts, no network, no Docker. |
| 2. Credentialed rollout on an official task | `python scripts/run_official.py --task address` | Snowflake + Airbyte (abctl) + Docker + ELT-Bench checkout |
| 3. One Tinker RL optimisation step | `python -m elt_rlvr.train max_steps=1` | `TINKER_API_KEY` |

### Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]" -e /path/to/tinker-cookbook
python scripts/build_mini_shop.py      # regenerates the local fixture (already committed)
pytest                                  # check 1: 25 tests, ~15 s
elt-rlvr demo-local --verbose           # one full local episode with the scripted oracle -> reward 1.0
elt-rlvr audit                          # reward matrix: 7 scripted policies x 3 reward modes
```

### Check 1: what the local test actually exercises

The fixture task `fixtures/mini_shop` has three source types (CSV, JSONL,
SQLite), two data models, and the edge cases that break naive SQL (a customer
with no orders, a NULL quantity, a tie-break, a flag defined against an
aggregate). Episodes run through tinker-cookbook's real
`AgentToolMessageEnv`, so tool dispatch, stop semantics and grading are the
same code the RL loop uses; only sampling is replaced by scripted policies:

```
reward_mode    policy              reward  el_frac  models_ok  col_frac
staged         oracle               1.000     1.00          2      1.00
staged         no_submit            0.000     1.00          2      1.00   # correct but never submitted
staged         el_only              0.200     1.00          0      0.00
staged         wrong_models         0.350     1.00          0      0.38   # easy columns right, aggregates wrong
staged         views_not_tables     0.200     1.00          0      0.00   # dbt default materialisation
staged         fake_load            0.000     0.00          2      1.00   # row counts forged, provenance fails
staged         append_twice         0.000     0.00          0      0.55   # sync run twice -> counts doubled
```

### Check 2: official task

See [docs/OFFICIAL.md](docs/OFFICIAL.md) for the Airbyte, Snowflake and Docker
setup. The environment code path is identical to the local one; only three
pluggable pieces change: `DockerSandbox` instead of `LocalSandbox`,
`AirbyteTerraformRuntime` instead of `LocalELRuntime`, `SnowflakeDestination`
instead of `DuckDBDestination`.

### Check 3: training

```bash
export TINKER_API_KEY=...
python -m elt_rlvr.train task_dirs='["fixtures/mini_shop"]' model_name=Qwen/Qwen3-8B \
    group_size=4 groups_per_batch=1 max_steps=1 log_path=runs/first-step
```

The entry point is a copy of `recipes/harbor_rl/train.py` with our dataset
builder; `tinker_cookbook.rl.train.main` does rollouts, GRPO-style advantages,
`forward_backward` and `optim_step`.

## Layout

```
src/elt_rlvr/
  task.py              TaskSpec: agent-visible half vs grader-only half
  destinations/        Destination + BaseConnector (ELT-Bench's evaluator interface): duckdb, snowflake
  el_runtime/          ELRuntime: local loader (credential-free) and Airbyte/Terraform (official)
  sandbox.py           LocalSandbox (subprocess) and DockerSandbox (elt-swe image)
  tools.py             the action space: bash, sql (read-only), run_el, submit
  grading.py           Stage 1 row counts + provenance, Stage 2 official comparator
  reward.py            binary / model_fraction / staged
  episode.py           one rollout: workspace + namespace + runtime + tools + grader closure
  env.py               EnvGroupBuilder / RLDataset / RLDatasetBuilder for tinker-cookbook
  train.py             Tinker RL entry point
  policies.py          scripted oracle and adversarial policies
  scripted.py          drive AgentToolMessageEnv with a scripted policy
  eltbench_compat.py   vendored ELT-Bench stage-2 comparator (+ parity test against upstream)
fixtures/mini_shop/    the credential-free task
tests/                 25 tests, no credentials
DESIGN.md              the design decisions
```

## Adding a warehouse

Implement `Destination.provision / connector / teardown` (plus `snapshot` for
transform-only starts) and a `BaseConnector` subclass with `_norm`, `_fqn`,
`list_tables`, `verify_schema`. `DuckDBDestination` is 110 lines. Nothing in
tools, grading, reward or the Tinker integration changes.
