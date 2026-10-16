"""Official-task episodes: ELT-Bench inputs + Docker sandbox + Airbyte/Terraform + a
real warehouse destination."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import yaml

from .destinations.base import Destination
from .el_runtime.airbyte_tf import AirbyteTerraformRuntime
from .episode import Episode, EpisodeConfig
from .sandbox import DockerSandbox
from .task import TaskSpec

AIRBYTE_AUTH_NOTE = """

## Authentication on this deployment

This Airbyte server authenticates API clients with application credentials, not
username/password. Configure the provider with the `client_id` and `client_secret`
values from `/workspace/config.yaml` (section `Airbyte.config`):

```hcl
provider "airbyte" {
  server_url    = "<Airbyte.config.server_url from config.yaml>"
  client_id     = "<Airbyte.config.client_id from config.yaml>"
  client_secret = "<Airbyte.config.client_secret from config.yaml>"
}
```

Sync jobs are triggered and monitored by the `run_el` tool; you do not need to call
the jobs API yourself.
"""


def _prepare_official_workspace(task: TaskSpec, handle, workdir: Path, runtime) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(task.agent_dir, workdir)
    cfg_path = workdir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    # Point the task at this rollout's namespace instead of the shared benchmark database.
    cfg[handle.destination.name] = {"config": dict(handle.agent_config)}
    # Airbyte >= 1.6 rejects basic auth on the public API. Give the agent the client
    # credentials the Terraform provider (>= 0.6) accepts, next to the upstream fields.
    ab = runtime._cred if hasattr(runtime, "_cred") else {}
    if ab.get("client_id") and "Airbyte" in cfg:
        cfg["Airbyte"]["config"]["client_id"] = ab["client_id"]
        cfg["Airbyte"]["config"]["client_secret"] = ab["client_secret"]
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    root = Path(task.extra["elt_bench_root"])
    helper = root / "setup" / "check_job_status.py"
    if helper.exists():
        shutil.copy(helper, workdir / "check_job_status.py")
    docs = workdir / "documentation"
    docs.mkdir(exist_ok=True)
    for name, content in runtime.agent_documentation().items():
        (docs / name).write_text(content)
    prov_doc = docs / "airbyte_Provider.md"
    if prov_doc.exists() and "client_id" not in prov_doc.read_text():
        prov_doc.write_text(prov_doc.read_text() + AIRBYTE_AUTH_NOTE)
    (workdir / ".dbt").mkdir(exist_ok=True)
    (workdir / ".dbt" / "profiles.yml").write_text(yaml.safe_dump(handle.dbt_profile, sort_keys=False))
    # Terraform provider lock/cache lives in the workspace so rollouts do not share state.
    (workdir / "elt").mkdir(exist_ok=True)


async def create_official_episode(task: TaskSpec, destination: Destination, rollouts_root: str | Path,
                                  cfg: EpisodeConfig | None = None, rollout_id: str | None = None,
                                  image: str = "elt-swe", network: str = "elt-docker_elt_network") -> Episode:
    cfg = cfg or EpisodeConfig()
    rollout_id = rollout_id or uuid.uuid4().hex[:8]
    workdir = Path(rollouts_root).resolve() / f"{task.name}-{rollout_id}"
    handle = await destination.provision(task.name, rollout_id)
    sandbox = DockerSandbox(workdir, image=image, network=network,
                            env={"DBT_PROFILES_DIR": "/workspace/.dbt", "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"})
    runtime = AirbyteTerraformRuntime(handle, workdir, sandbox,
                                      sync_delay_seconds={"databricks": 60, "redshift": 20}.get(destination.name, 0))
    _prepare_official_workspace(task, handle, workdir, runtime)
    return Episode(task, handle, sandbox, runtime, workdir, cfg, rollout_id)
