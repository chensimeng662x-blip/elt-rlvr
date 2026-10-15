# Running an official ELT-Bench task

Everything below is the upstream ELT-Bench setup; this environment adds only the
per-rollout namespace and the host-side `run_el` runtime.

## 1. Infrastructure (once)

```bash
git clone -b elt_bench++ https://github.com/uiuc-kang-lab/ELT-Bench.git
cd ELT-Bench
# Airbyte OSS via abctl (needs Docker with >= 8 GB RAM); follow upstream README:
#   abctl local install ; abctl local credentials
#   import setup/elt_bench.yaml in the Builder UI, record workspace id + custom-api definition id
#   fill setup/airbyte/airbyte_credential.json
# Snowflake: run setup/destination/setup.sql, fill setup/destination/snowflake_credential.json
cd setup && bash elt_setup.sh            # downloads data, starts source containers, writes inputs/
docker build -t elt-swe ../agents/SWE-agent/docker/elt-swe
```

## 2. Credentials for this environment

```bash
export ELT_RLVR_SNOWFLAKE_CREDENTIAL=/path/to/snowflake_credential.json   # admin account used for provisioning + grading
export ELT_RLVR_AIRBYTE_CREDENTIAL=/path/to/airbyte_credential.json       # {"username":..., "password":...}
export ELT_RLVR_AIRBYTE_URL=http://localhost:8000/api/public/v1
```

Add `agent_user` / `agent_password` / `agent_role` to the Snowflake JSON if they
differ from the `AIRBYTE_USER` / `AIRBYTE_ROLE` created by `setup.sql`. Credential
files are never copied into the workspace; the agent sees only the values that
ELT-Bench's own `config.yaml` already exposes.

## 3. One rollout, scripted or sampled

```bash
python scripts/run_official.py --elt-bench-root ../ELT-Bench --task address --print-grade
python -m elt_rlvr.train mode=eltbench elt_bench_root=../ELT-Bench destination=snowflake \
    task_dirs='["address"]' group_size=2 max_steps=1
```

## Isolation model

- Each rollout gets database `ELT_<task>_<rollout>` with schema `AIRBYTE_SCHEMA`
  owned by the Airbyte role, a Docker container on the benchmark network, and a
  Terraform state file inside its own workspace. Teardown drops the database and
  the container.
- Grading uses the admin connection from the host. Ground truth never enters the
  container.
- Provenance: every table credited in Stage 1 must have been produced by an Airbyte
  job that `run_el` triggered and observed succeeding, and its content fingerprint
  must still match what that job loaded.
