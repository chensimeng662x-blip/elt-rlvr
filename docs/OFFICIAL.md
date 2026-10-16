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

## Deviations from the upstream setup (found while deploying on 2026-10-17, macOS arm64, Airbyte 2.4.0)

1. **Agent image.** Upstream `agents/SWE-agent/docker/elt-swe/Dockerfile` installs
   `software-properties-common`, which no longer exists on Debian trixie (the current
   `python:3.11` base). `docker/elt-swe/Dockerfile` here pins `python:3.11-bookworm`
   with the same tool set.
2. **LocalStack.** `localstack/localstack:latest` now exits with "pro features require a
   license". `docker/elt-sources.override.yml` (drop it into `elt-docker/` as
   `docker-compose.override.yml`) pins the community image `3.8.1` with `SERVICES=s3`,
   which is all ELT-Bench uses.
3. **Custom API connector.** The Builder manifest `setup/elt_snowflake.yaml` declares
   CDK `6.33.4`; Airbyte 2.4.0's `declarative_manifest_image_version` table only knows
   majors 0 to 4, so both the UI import and the public API return a 409. Registering the
   version makes the publish succeed:

   ```sql
   -- inside the abctl cluster: kubectl exec -n airbyte-abctl airbyte-db-0 -- psql -U airbyte -d db-airbyte
   insert into declarative_manifest_image_version (major_version, image_version, image_sha, created_at, updated_at)
   values (6, '6.33.4', '<digest of airbyte/source-declarative-manifest:6.33.4>', now(), now());
   ```

   then `POST /api/public/v1/workspaces/{id}/definitions/declarative_sources` with
   `{"name": "elt_bench", "manifest": <yaml>}`. The returned `id` is the
   `api_definition_id` for `setup/airbyte/airbyte_credential.json`. `scripts/airbyte_bootstrap.py`
   automates the first-run setup, workspace lookup and this publish.
4. **Connector drift vs. the upstream ground truth.** `example/retails/main.tf` was written for
   2025 connector versions. With the connectors Airbyte 2.4 installs:
   - `source-postgres` 3.8 requires `replication_method` and `tunnel_method` (both already
     listed as required in ELT-Bench's `documentation/source_postgres.md`) and defaults to
     `ssl_mode = require`; the benchmark's Postgres container has no SSL, so use
     `ssl_mode = { disable = {} }`. `scripts/official_oracle.py` applies all three.
   - `source-mongodb-v2` 2.x renamed `database` to a `databases` list, which contradicts the
     agent-facing documentation. Pin it back to the last 1.x: `POST /api/v1/source_definitions/update`
     with `{"sourceDefinitionId": "b2e713cd-cc36-4c0a-b5bd-b47cb8a0561e", "dockerImageTag": "1.5.19", "workspaceId": ...}`.
5. **Memory.** On a 16 GB Docker VM the MongoDB container's default WiredTiger cache (~50% of
   RAM) left the Airbyte Postgres source OOM-killed mid-sync ("No exit code file found ...
   Container may have been killed"). The override caps it at 1 GB.
6. **Job status semantics.** Airbyte reports `incomplete` for a job whose attempt failed and
   is waiting for a retry; `run_el` treats only `succeeded` / `failed` / `cancelled` as terminal.
