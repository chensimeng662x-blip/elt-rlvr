"""Tinker RL entry point (mirrors ``tinker_cookbook/recipes/harbor_rl/train.py``).

Example (credential-free fixture, one optimisation step):

    python -m elt_rlvr.train task_dirs='["fixtures/mini_shop"]' max_steps=1 \
        model_name=Qwen/Qwen3-8B group_size=4 groups_per_batch=1
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from .env import ELTDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    task_dirs: list[str] = chz.field(default_factory=lambda: ["fixtures/mini_shop"])
    mode: str = "local"
    elt_bench_root: str | None = None
    destination: str = "duckdb"

    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None
    lora_rank: int = 32
    load_checkpoint_path: str | None = None
    max_tokens: int = 2048
    temperature: float = 1.0

    group_size: int = 4
    groups_per_batch: int = 1
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    max_steps: int | None = 1
    max_steps_off_policy: int | None = None

    max_turns: int = 40
    max_trajectory_tokens: int = 28 * 1024  # Qwen3-8B on Tinker has a 32K window; leave room for max_tokens
    max_tool_calls: int = 60
    command_timeout: int = 600
    max_output_chars: int = 8 * 1024
    reward_mode: str = "staged"
    el_weight: float = 0.2
    column_credit: float = 0.5
    require_provenance: bool = True
    start_stage: str = "full"  # or "transform" (curriculum stage 1: Stage 1 preloaded)

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 0
    save_every: int = 1
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(c: CLIConfig) -> None:
    renderer_name = c.renderer_name or model_info.get_recommended_renderer_name(c.model_name)
    run_name = f"elt-{c.mode}-{c.model_name.replace('/', '-')}-g{c.group_size}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    log_path = c.log_path or cli_utils.runs_path("elt_rlvr", run_name)

    dataset_builder = ELTDatasetBuilder(
        task_dirs=c.task_dirs, mode=c.mode, elt_bench_root=c.elt_bench_root, destination=c.destination,
        model_name=c.model_name, renderer_name=renderer_name, group_size=c.group_size,
        groups_per_batch=c.groups_per_batch, max_turns=c.max_turns, max_trajectory_tokens=c.max_trajectory_tokens,
        max_tool_calls=c.max_tool_calls, max_generation_tokens=c.max_tokens, reward_mode=c.reward_mode,
        el_weight=c.el_weight, column_credit=c.column_credit, command_timeout=c.command_timeout, max_output_chars=c.max_output_chars,
        require_provenance=c.require_provenance, start_stage=c.start_stage,
    )
    config = Config(
        learning_rate=c.learning_rate, dataset_builder=dataset_builder, model_name=c.model_name,
        recipe_name="elt_rlvr", lora_rank=c.lora_rank, max_tokens=c.max_tokens, temperature=c.temperature,
        wandb_project=c.wandb_project, wandb_name=c.wandb_name or run_name, log_path=log_path,
        base_url=c.base_url, load_checkpoint_path=c.load_checkpoint_path, kl_penalty_coef=c.kl_penalty_coef,
        num_substeps=c.num_substeps, eval_every=c.eval_every, save_every=c.save_every,
        async_config=AsyncConfig(max_steps_off_policy=c.max_steps_off_policy, groups_per_batch=c.groups_per_batch)
        if c.max_steps_off_policy is not None else None,
        max_steps=c.max_steps,
        rollout_error_tolerance=True,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=c.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
