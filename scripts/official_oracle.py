"""Credentialed end-to-end check on the official ELT-Bench task ``retails``.

Drives one real episode through the environment's own tools with ELT-Bench's
ground-truth pipeline (``example/retails/main.tf`` + ``customers.sql`` /
``nations.sql``): Terraform against the local Airbyte, syncs into a per-rollout
Snowflake database, dbt in the elt-swe container, grading by warehouse state.
A correct run must end with el_complete=True, 2/2 models correct, reward 1.0.

    source credentials/env.sh
    python scripts/official_oracle.py --elt-bench-root ../ELT-Bench [--keep]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import yaml

from elt_rlvr.destinations.snowflake_dest import SnowflakeDestination
from elt_rlvr.episode import EpisodeConfig
from elt_rlvr.official import create_official_episode
from elt_rlvr.task import load_eltbench_task

DBT_PROJECT = """name: elt_models
version: '1.0'
config-version: 2
profile: elt
model-paths: ["models"]
models:
  elt_models:
    +materialized: table
"""


def render_main_tf(template: str, cfg: dict, physical_db: str) -> str:
    ab = cfg["Airbyte"]["config"]
    sf = cfg["snowflake"]["config"]
    tf = template
    # provider block: client credentials (Airbyte >= 1.6 public API)
    tf = re.sub(r'provider "airbyte" \{.*?\n\}', 'provider "airbyte" {\n  server_url    = "%s"\n  client_id     = "%s"\n  client_secret = "%s"\n}' % (
        ab["server_url"], ab["client_id"], ab["client_secret"]), tf, flags=re.S)
    tf = tf.replace("<workspace_id>", ab["workspace_id"]).replace("<custom_api_definition_id>", ab["custom_api_definition_id"])
    tf = tf.replace("<snowflake_host>", f"{sf['account']}.snowflakecomputing.com")
    tf = tf.replace('database  = "retails"', f'database  = "{physical_db}"')
    tf = tf.replace('password = "Snowflake@123"', f'password = "{sf["password"]}"')
    # The upstream example predates source-postgres 3.8 (installed by Airbyte 2.4), which
    # requires replication_method and tunnel_method. ELT-Bench's own documentation already
    # lists both as required, so the agent-facing contract is unchanged.
    pg_old = '    schemas  = ["public"]\n  }'
    # ssl_mode: source-postgres 3.8 defaults to "require"; the benchmark's Postgres container has no SSL.
    pg_new = ('    schemas  = ["public"]\n'
              '    replication_method = {\n      scan_changes_with_user_defined_cursor = {}\n    }\n'
              '    ssl_mode = {\n      disable = {}\n    }\n'
              '    tunnel_method = {\n      no_tunnel = {}\n    }\n  }')
    assert pg_old in tf, "postgres block not found in template"
    tf = tf.replace(pg_old, pg_new)
    return tf


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elt-bench-root", required=True)
    ap.add_argument("--task", default="retails")
    ap.add_argument("--rollouts-root", default="runs/rollouts")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-el", action="store_true", help="assume raw tables are already loaded (debug transform stage)")
    args = ap.parse_args()
    root = Path(args.elt_bench_root).resolve()

    task = load_eltbench_task(root, args.task, destination="snowflake")
    dest = SnowflakeDestination.from_env()
    cfg = EpisodeConfig()
    ep = await create_official_episode(task, dest, args.rollouts_root, cfg)
    ep.el_runtime.sync_timeout = 5400
    t0 = time.time()
    print("workspace:", ep.workdir)
    print("snowflake database:", ep.handle.extra["physical_database"])
    try:
        config = yaml.safe_load((ep.workdir / "config.yaml").read_text())
        physical_db = ep.handle.extra["physical_database"]
        tf = render_main_tf((root / "example" / args.task / "main.tf").read_text(), config, physical_db)
        (ep.workdir / "elt" / "main.tf").write_text(tf)
        print("wrote elt/main.tf (%d lines)" % tf.count("\n"))

        if not args.skip_el:
            r = await ep.tools.run_el.run(type("I", (), {"arguments": {}, "call_id": "oracle"})())
            out = json.loads(r.messages[0]["content"])
            print("run_el ->", json.dumps(out, indent=1)[:3000])
            print("elapsed after EL: %.0fs" % (time.time() - t0))

        dbt_dir = ep.workdir / "elt" / "dbt"
        (dbt_dir / "models").mkdir(parents=True, exist_ok=True)
        (dbt_dir / "dbt_project.yml").write_text(DBT_PROJECT)
        for sql in (root / "example" / args.task).glob("*.sql"):
            body = sql.read_text().replace(f"{args.task}.airbyte_schema.", "AIRBYTE_SCHEMA.")
            (dbt_dir / "models" / sql.name).write_text(body)
        r = await ep.tools.bash.run(type("I", (), {"arguments": {"command": "cd /workspace/elt/dbt && dbt run --no-use-colors 2>&1 | tail -n 20"}, "call_id": "oracle"})())
        print("dbt run ->", json.loads(r.messages[0]["content"])["stdout"][-2500:])

        await ep.tools.submit.run(type("I", (), {"arguments": {}, "call_id": "oracle"})())
        reward, metrics = await ep.reward_fn([])
        print("REWARD:", reward)
        print(json.dumps(ep.last_report.summary(), indent=1))
        print("total elapsed: %.0fs" % (time.time() - t0))
        Path("runs").mkdir(exist_ok=True)
        Path(f"runs/official_oracle_{args.task}.json").write_text(json.dumps({"reward": reward, "metrics": metrics, "grade": ep.last_report.summary(), "elapsed_s": time.time() - t0}, indent=1))
    finally:
        if args.keep:
            print("kept workspace and database (clean up with: python scripts/official_oracle.py ... without --keep, or drop the database manually)")
        else:
            await ep.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
