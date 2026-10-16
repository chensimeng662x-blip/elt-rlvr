# Validation status

| Item | Status | Evidence |
|---|---|---|
| Local integration test (check 1) | **verified** | `pytest`: 25 passed (18 + 7 parity tests that need `ELT_BENCH_ROOT`) |
| Comparator parity with upstream `eva_stage2.py` | **verified** | `tests/test_comparator_parity.py` with `ELT_BENCH_ROOT=~/Projects/ELT-Bench` |
| Scripted audit (7 policies x 3 reward modes) | **verified** | `elt-rlvr audit`, table in README |
| Renderer + tool schema for Qwen3-8B | **verified offline** | `ELTEnvGroupBuilder.make_envs()` renders the initial prompt with the Qwen3 renderer |
| Official rollout on Snowflake + Airbyte (check 2) | pending credentials | code written against upstream connector / `check_job_status.py` / `trigger_job.md` |
| Tinker optimisation step (check 3) | **verified** (2026-10-17) | `runs/step-full`: Qwen/Qwen3-8B, LoRA 32, group of 4 on `mini_shop`, 40-turn budget. Rewards 0.2 / 0.2 / 0 / 0 (two rollouts completed Stage 1 and submitted, two hit max_turns), `frac_mixed = 1.0`, `optim_step` executed, checkpoint written. First attempt (`runs/first-step`) had all-zero rewards; the transcripts showed two environment faults (relative paths in `config.yaml`, BSD `sed -i` in the local sandbox), fixed before the second run. |

## Training step details (check 3)

| run | stage | turns | rewards (group of 4) | group | notes |
|---|---|---|---|---|---|
| `runs/first-step` | full | 25 | 0, 0, 0, 0 | all-bad | env faults: relative paths, BSD sed; 8 turns lost to `sed -i` |
| `runs/step-full` | full | 40 | 0.2, 0.2, 0, 0 | mixed | 2/4 completed EL via `run_el` and submitted; 0/4 produced a correct model |
| `runs/step-transform` | transform-only | 40 | see log | | curriculum stage: Stage 1 preloaded through the EL runtime |

`metrics.jsonl` and `iteration_000000/train_rollout_summaries.jsonl` in each run directory hold the per-trajectory grades (`grade/el_fraction`, `grade/models_correct`, ...) and the full tool-call transcripts.
