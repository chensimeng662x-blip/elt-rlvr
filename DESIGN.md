# ELT-Bench RLVR environment: design

One page of decisions. Code: this repository. Benchmark: ELT-Bench (Jin, Zhu, Kang 2025), branch `elt_bench++`.

## 1. The task as an RL episode

An ELT-Bench task is a specification (`config.yaml`, `data_model.yaml`, `schemas/`, `documentation/`), five source systems, Airbyte for extract/load, dbt for transform, and a warehouse. The episode is **stateful**: files the policy writes, sync jobs it triggers, tables dbt creates all persist across turns. State lives in three places: the workspace filesystem, the Airbyte job history, and the warehouse namespace. Terminal conditions: the policy calls `submit`, or it stops calling tools, or a budget (turns, tool calls, trajectory tokens, wall clock) runs out.

Action space is four tools, identical for every warehouse: `bash` (workspace shell), `sql` (read-only query on the rollout's own namespace), `run_el` (host-mediated extraction and loading from the configuration the policy wrote), `submit`. Prompts are ELT-Bench's own agent prompts with these tool names.

Everything a rollout owns is per rollout: a workspace directory, a sandbox (subprocess locally, Docker container officially), a warehouse namespace (DuckDB file / Snowflake database), an EL runtime holding that rollout's provenance. A group is one task times `group_size` such episodes.

## 2. Reuse

- Tinker side: `tinker_cookbook.tool_use.build_agent_tool_env` provides the multi-turn tool loop, parse-error policy, token/turn/tool-call budgets and stop reasons; `rl.train.main` provides rollouts, GRPO-style advantages, `forward_backward`, `optim_step`. Our code is an `EnvGroupBuilder`, an `RLDataset`, four tools and a reward closure. The training entry point is `recipes/harbor_rl/train.py` with the dataset builder swapped.
- ELT-Bench side: grading calls the upstream stage-1 logic (row counts via `BaseConnector.table_size`) and the upstream stage-2 comparator (`check_corretness`: sort by unique key, numeric coercion with rtol 1e-2, case-insensitive strings, extra columns allowed). The comparator is vendored unchanged and a parity test runs it against the upstream file.

## 3. Destination-specific, not destination-bound

`Destination` (provision / connector / teardown / snapshot) plus a `BaseConnector` subclass is the whole warehouse surface. `DuckDBDestination` (local, credential-free, 110 lines) and `SnowflakeDestination` (official) implement it. The Snowflake connector maps the logical task name to a per-rollout database so upstream evaluation SQL (`<task>.<table>`) resolves unchanged. Databricks or Redshift follow the upstream connectors in `evaluation/db_connectors.py`.

## 4. Reward

Definition of a correct pipeline is the benchmark's: every raw table present with exactly the expected row count, and every data model matching the ground truth under the official comparator. Verification is by execution against the warehouse, never by inspecting the transcript or the workspace.

Outcome reward only. Three modes, all computed from a `GradeReport`:

- `binary`: 1 iff fully correct. Matches the benchmark; useless for learning early on because success is near zero and a group of zeros has no advantage.
- `staged` (default): `w_el * el_fraction + [el complete] * (1 - w_el) * mean_m(score_m)`, with `score_m = 1` on a match else `column_credit * matched_columns / gold_columns`. Models are only ever graded after a complete Stage 1 (upstream semantics). Column credit is capped at `column_credit = 0.5`, so no amount of partial columns reaches a real match.
- `model_fraction`: `staged` without column credit.

Why not process rewards (terraform applied, dbt ran, tool-call counts): they are proxies the policy can satisfy without moving data, and ELT-Bench's own error analysis shows the failure mass is in semantic SQL errors and misconfiguration, which only the warehouse state reveals. The single process-shaped term is negative: a parse failure or context overflow costs -0.1, and any limit-stopped rollout is clamped to <= 0 (`TerminationRewardPolicy.zero_reward_on_limit`), so running out the clock is never rewarded. Unsubmitted episodes score 0.

## 5. Reward hacking: holes and plugs

| Hole | Plug | Test |
|---|---|---|
| Forge raw tables with `INSERT` / a Python script so row counts match | Provenance: Stage 1 credit requires the table to carry Airbyte metadata columns **and** a content fingerprint equal to the one `run_el` recorded from the warehouse right after the sanctioned load | `fake_load` -> 0 |
| Read the ground truth | Grader files never enter the workspace or container; grading runs host-side with its own connection | layout |
| Tamper with grading | Agent credentials own only their namespace; grader uses a separate admin connection; `sql` tool is read-only by statement whitelist and, on DuckDB, by a read-only connection | `tools.py` |
| Collect column credit from trivial columns | Credit capped at 0.5 per model; models only graded after complete EL; audit shows `wrong_models` = 0.35 vs oracle 1.0 | `wrong_models` |
| Skip the hard part and submit early | `el_only` = `w_el` = 0.2 and nothing more | `el_only` |
| Re-run syncs in append mode | Exact row-count equality, as upstream | `append_twice` -> 0 |
| Views instead of tables | Only `BASE TABLE` counts, as upstream | `views_not_tables` |
| Cross-rollout reuse | Per-rollout namespace, teardown after grading | design |
| Stall | turn / tool-call / token / wall-clock budgets with reward clamp | preset |

Residual: with a shared Airbyte role on Snowflake the agent could read (not be credited for) another rollout's database; per-rollout roles are a direct extension of `provision`.

## 6. Efficiency

- Transform-only starts: `Destination.snapshot` (DuckDB file copy; Snowflake zero-copy `CLONE`) lets a curriculum stage begin from a loaded warehouse and skip the 5 to 15 minute Airbyte sync. This also raises the non-zero-reward rate early in training.
- The local EL runtime replaces Airbyte for most training rollouts; the official runtime is for evaluation and late-stage training.
- Rollouts in a group run concurrently; tool execution within a rollout is sequential (shared filesystem and warehouse). Terraform runs with `-parallelism=1` (Airbyte 0.6.5 provider credential-cache race).
- Tool output capped at 16 KB; system prompt and documentation form a stable prefix (Tinker's 80% cached-prefill discount).
- Grading cost is one `COUNT(*)` per raw table, one fingerprint per raw table, one fetch per model; ground-truth CSVs are read once.
- Dataset cycles through tasks so batches rotate rather than starve.

## 7. Known limitation worth stating

On official tasks the base-model success rate is near zero (best published agent: 3.9% of models), so even `staged` groups will often be all-zero. The levers, in order: transform-only snapshots, task selection by table count, larger `group_size`, a stronger base model. The one required optimisation step runs regardless; a useful learning curve needs the curriculum.
