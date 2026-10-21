# ELT-Bench RLVR environment: design

One page of decisions. Code: https://github.com/chensimeng662x-blip/elt-rlvr. Benchmark: ELT-Bench by Jin, Zhu and Kang, 2025, branch `elt_bench++`.

## 1. The task as an RL episode

An ELT-Bench task has five parts: a specification, five source systems, Airbyte for extraction and loading, dbt for transformation, and a warehouse. The specification is `config.yaml`, `data_model.yaml`, `schemas/` and `documentation/`.

The episode is **stateful**. Files the policy writes, sync jobs it triggers and tables dbt creates all persist across turns. State lives in three places: the workspace filesystem, the Airbyte job history and the warehouse namespace. The episode ends when the policy calls `submit`, when it stops calling tools, or when a budget runs out. Budgets cover turns, tool calls, trajectory tokens and wall clock time.

The action space is four tools, identical for every warehouse:

- `bash` runs shell commands in the workspace.
- `sql` runs read-only queries on the rollout's own namespace.
- `run_el` runs extraction and loading on the host, from the configuration the policy wrote.
- `submit` ends the episode.

Prompts are ELT-Bench's own agent prompts with these tool names.

Everything a rollout touches belongs to that rollout alone: a workspace directory, a sandbox, a warehouse namespace and an EL runtime that holds the rollout's provenance. Locally the sandbox is a subprocess and the namespace is a DuckDB file. Officially they are a Docker container and a Snowflake database. A group is one task times `group_size` such episodes.

## 2. Reuse

- Tinker side. `tinker_cookbook.tool_use.build_agent_tool_env` provides the multi-turn tool loop, the parse-error policy, the budgets and the stop reasons. `rl.train.main` provides rollouts, GRPO-style advantages, `forward_backward` and `optim_step`. Our code is an `EnvGroupBuilder`, an `RLDataset`, four tools and a reward function. The training entry point is `recipes/harbor_rl/train.py` with the dataset builder swapped.
- ELT-Bench side. Stage 1 grading reuses the upstream row-count logic in `BaseConnector.table_size`. Stage 2 grading reuses the upstream comparator `check_corretness`. It sorts by unique key, compares numbers with a relative tolerance of 1e-2, compares strings case-insensitively and allows extra columns. The comparator is vendored unchanged, and a parity test checks it against the upstream file.

## 3. Destination-specific, not destination-bound

The whole warehouse surface is one `Destination` class plus one `BaseConnector` subclass. `Destination` provides provision, connector, teardown and snapshot. `DuckDBDestination` is local, needs no credentials and is about 110 lines. `SnowflakeDestination` is the official one. The Snowflake connector maps the logical task name to a per-rollout database, so upstream evaluation SQL written as `task.table` resolves unchanged. Databricks or Redshift would follow the upstream connectors in `evaluation/db_connectors.py`.

## 4. Reward

A correct pipeline is defined as in the benchmark. Every raw table is present with exactly the expected row count, and every data model matches the ground truth under the official comparator. Correctness is checked by querying the warehouse. The grader never reads the transcript or the workspace.

The reward is outcome only. There are three modes, all computed from a `GradeReport`:

- `binary` gives 1 only if everything is correct. It matches the benchmark, but it is useless early in training. Success is near zero, and a group of zeros has no advantage.
- `staged` is the default. Extraction and loading earn `w_el * el_fraction`. Once Stage 1 is complete, the data models share the remaining `1 - w_el`. A matching model scores 1. A partly correct model scores `column_credit * matched_columns / gold_columns`. Models are graded only after Stage 1 is complete, as upstream. Column credit is capped at 0.5, so partial columns can never add up to a real match.
- `model_fraction` is `staged` without column credit.

Process rewards are left out on purpose. Signals such as "terraform applied", "dbt ran" or tool-call counts are proxies the policy can satisfy without moving any data. ELT-Bench's own error analysis also shows that most failures are SQL logic errors and misconfiguration, which only the warehouse state reveals.

The only process-shaped term is negative. A parse failure or a context overflow costs 0.1. Any rollout stopped by a budget is capped at a reward of 0, so running out the clock is never rewarded. Unsubmitted episodes score 0.

## 5. Reward hacking: holes and plugs

| Hole | Plug | Test |
|---|---|---|
| Forge raw tables with `INSERT` or a Python script so row counts match | Provenance. Stage 1 credit needs Airbyte metadata columns and a content fingerprint equal to the one `run_el` recorded right after the real load | `fake_load` scores 0 |
| Read the ground truth | Grader files never enter the workspace or container. Grading runs on the host with its own connection | layout |
| Tamper with grading | Agent credentials own only their namespace. The grader uses a separate admin connection. The `sql` tool accepts read-only statements only | `tools.py` |
| Collect column credit from trivial columns | Credit is capped at 0.5 per model and models are graded only after complete EL. The audit shows `wrong_models` at 0.35 against the oracle's 1.0 | `wrong_models` |
| Skip the hard part and submit early | This earns `w_el`, which is 0.2, and nothing more | `el_only` |
| Re-run syncs in append mode | Row counts must match exactly, as upstream | `append_twice` scores 0 |
| Views instead of tables | Only `BASE TABLE` counts, as upstream | `views_not_tables` |
| Reuse another rollout's tables | Each rollout has its own namespace, torn down after grading | design |
| Stall | Turn, tool-call, token and wall clock budgets, with the reward cap | preset |

One gap remains. On Snowflake all rollouts share one Airbyte role, so an agent could read another rollout's database, though it gets no credit for it. Giving each rollout its own role is a direct extension of `provision`.

## 6. Efficiency

- Transform-only starts. `Destination.snapshot` copies a loaded warehouse, as a DuckDB file copy or a Snowflake zero-copy `CLONE`. A curriculum stage can start from it and skip the 5 to 15 minute Airbyte sync. This also raises the share of non-zero rewards early in training.
- The local EL runtime replaces Airbyte for most training rollouts. The official runtime is for evaluation and late-stage training.
- Rollouts in a group run in parallel. Tools inside one rollout run in order, because they share a filesystem and a warehouse. Terraform runs with `-parallelism=1` to avoid a credential-cache race in the Airbyte 0.6.5 provider.
- Tool output is capped at 16 KB. The system prompt and documentation form a stable prefix, which Tinker bills at an 80% discount.
- Grading costs one `COUNT(*)` and one fingerprint per raw table and one fetch per model. Ground-truth CSVs are read once.
- The dataset cycles through tasks, so every task gets its turn in the batches.

## 7. Known limitation

On official tasks the base model's success rate is near zero. The best published agent gets 3.9% of models right. So even `staged` groups will often be all zero. The levers, in order, are transform-only snapshots, picking tasks with fewer tables, a larger `group_size` and a stronger base model. The one required optimisation step runs regardless. A useful learning curve needs the curriculum.

## 8. Validation, 2026-10-17

| Check | Result |
|---|---|
| 1. Credential-free local test | 26 tests pass, including parity with upstream `eva_stage2.py`. 7 scripted policies across 3 reward modes behave as designed: oracle 1.0, forged load 0, append twice 0, unsubmitted 0 |
| 2. Official task on Snowflake and Airbyte | `retails`: 8 of 8 raw tables at the expected counts with provenance, 2 of 2 models match the official comparator, reward 1.0, 568 seconds. Three earlier attempts failed on connector drift, the Postgres SSL default, a REST API that ignored pagination and an out-of-memory connector pod. Each failure showed up as a row-count or provenance miss in the grade report |
| 3. Tinker RL step | Qwen3-8B, LoRA rank 32, a group of 4, 40 turns. Rewards were 0.2, 0.2, 0 and 0: two rollouts completed extraction and loading through `run_el` and submitted. The group is mixed, so advantages are non-zero, and `optim_step` ran and saved a checkpoint. A first attempt scored 0 on all four. Its transcripts showed two environment faults, relative paths in `config.yaml` and BSD `sed` in the local sandbox, which were fixed before the second run |

The all-zero first step is the case section 7 predicts. The same zero reward covered both environment faults and model weakness. Reading the transcripts is what told them apart.
