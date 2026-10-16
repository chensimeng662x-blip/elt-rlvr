# Validation status

| Item | Status | Evidence |
|---|---|---|
| Local integration test (check 1) | **verified** | `pytest`: 25 passed (18 + 7 parity tests that need `ELT_BENCH_ROOT`) |
| Comparator parity with upstream `eva_stage2.py` | **verified** | `tests/test_comparator_parity.py` with `ELT_BENCH_ROOT=~/Projects/ELT-Bench` |
| Scripted audit (7 policies x 3 reward modes) | **verified** | `elt-rlvr audit`, table in README |
| Renderer + tool schema for Qwen3-8B | **verified offline** | `ELTEnvGroupBuilder.make_envs()` renders the initial prompt with the Qwen3 renderer |
| Official rollout on Snowflake + Airbyte (check 2) | **verified** (2026-10-17) | `scripts/official_oracle.py` on task `retails`: ELT-Bench's ground-truth pipeline driven through the environment's own tools (Terraform via `run_el` against local Airbyte 2.4, 8 streams from Postgres/MongoDB/S3/REST/JSONL into a per-rollout Snowflake database, dbt in the `elt-swe` container). All 8 raw tables at expected row counts with provenance, 2/2 data models match the official comparator, **reward 1.0**, 568 s wall clock (419 s extraction/loading). Report: `runs/official_oracle_retails.json`. Three earlier attempts failed on infrastructure drift (see docs/OFFICIAL.md, items 4 to 8), each caught by the grader. |
| Tinker optimisation step (check 3) | **verified** (2026-10-17) | `runs/step-full`: Qwen/Qwen3-8B, LoRA 32, group of 4 on `mini_shop`, 40-turn budget. Rewards 0.2 / 0.2 / 0 / 0 (two rollouts completed Stage 1 and submitted, two hit max_turns), `frac_mixed = 1.0`, `optim_step` executed, checkpoint written. First attempt (`runs/first-step`) had all-zero rewards; the transcripts showed two environment faults (relative paths in `config.yaml`, BSD `sed -i` in the local sandbox), fixed before the second run. |

## Training step details (check 3)

| run | stage | turns | rewards (group of 4) | group | notes |
|---|---|---|---|---|---|
| `runs/first-step` | full | 25 | 0, 0, 0, 0 | all-bad | env faults: relative paths, BSD sed; 8 turns lost to `sed -i` |
| `runs/step-full` | full | 40 | 0.2, 0.2, 0, 0 | mixed | 2/4 completed EL via `run_el` and submitted; 0/4 produced a correct model |
| `runs/step-transform` | transform-only | 40 | 0, 0, 0, -0.1 | mixed (by the overflow penalty only) | Stage 1 preloaded; Qwen3-8B still failed to produce a valid `dbt_project.yml` within 40 turns (confused with `profiles.yml`, hallucinated a BigQuery profile); one parse error, one context overflow. Model weakness, not an environment fault: transcripts show dbt's errors reaching the model each time. |

`metrics.jsonl` and `iteration_000000/train_rollout_summaries.jsonl` in each run directory hold the per-trajectory grades (`grade/el_fraction`, `grade/models_correct`, ...) and the full tool-call transcripts.

## Official run details (check 2)

| attempt | result | cause | fix |
|---|---|---|---|
| 1 | terraform apply 422 | source-postgres 3.8 requires `replication_method`/`tunnel_method`; source-mongodb-v2 2.x requires `databases` | add fields; pin mongodb connector to 1.5.19 |
| 2 | Postgres connection: "server does not support SSL" | connector default `ssl_mode=require` | `ssl_mode = disable` |
| 3 | reward 0.1: `part` 73.8 M rows, `customer` 2x, `lineitem`/`orders` > expected | REST API ignores `limit`/`offset` (declarative paginator never stops); MongoDB cache starved connector pods, failed attempts + `full_refresh_append` duplicate rows | API honours pagination; MongoDB cache capped at 1 GB |
| 4 | **reward 1.0** | | |

Every failure above surfaced as a Stage 1 row-count or provenance miss in the grade report, which is the behaviour the reward is supposed to have.
