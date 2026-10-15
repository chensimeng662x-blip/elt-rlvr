# Validation status

| Item | Status | Evidence |
|---|---|---|
| Local integration test (check 1) | **verified** | `pytest`: 25 passed (18 + 7 parity tests that need `ELT_BENCH_ROOT`) |
| Comparator parity with upstream `eva_stage2.py` | **verified** | `tests/test_comparator_parity.py` with `ELT_BENCH_ROOT=~/Projects/ELT-Bench` |
| Scripted audit (7 policies x 3 reward modes) | **verified** | `elt-rlvr audit`, table in README |
| Renderer + tool schema for Qwen3-8B | **verified offline** | `ELTEnvGroupBuilder.make_envs()` renders the initial prompt with the Qwen3 renderer |
| Official rollout on Snowflake + Airbyte (check 2) | pending credentials | code written against upstream connector / `check_job_status.py` / `trigger_job.md` |
| Tinker optimisation step (check 3) | pending `TINKER_API_KEY` | `python -m elt_rlvr.train max_steps=1` |
