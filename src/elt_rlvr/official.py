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


def _prepare_official_workspace(task: TaskSpec, handle, workdir: Path, runtime) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(task.agent_dir, workdir)
    cfg_path = workdir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    # Point the task at this rollout's namespace instead of the shared benchmark database.
    cfg[handle.destination.name] = {"config": dict(handle.agent_config)}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    root = Path(task.extra["elt_bench_root"])
    helper = root / "setup" / "check_job_status.py"
    if helper.exists():
        shutil.copy(helper, workdir / "check_job_status.py")
    docs = workdir / "documentation"
    docs.mkdir(exist_ok=True)
    for name, content in runtime.agent_documentation().items():
        (docs / name).write_text(content)
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
