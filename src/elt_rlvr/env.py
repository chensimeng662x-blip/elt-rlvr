"""Tinker integration: ``EnvGroupBuilder`` / ``RLDataset`` / ``RLDatasetBuilder``.

Each group = one task x ``group_size`` isolated episodes (own workspace, own
warehouse namespace, own EL runtime). Episodes are wrapped with
``tinker_cookbook.tool_use.build_agent_tool_env`` so parse handling, turn and
token budgets, and stop reasons are the cookbook's, not ours.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from pathlib import Path

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits, TerminationRewardPolicy
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from .destinations.base import Destination
from .destinations.duckdb_dest import DuckDBDestination
from .episode import Episode, EpisodeConfig, create_local_episode
from .prompts import SYSTEM_PROMPT, TASK_INSTRUCTION, TASK_INSTRUCTION_TRANSFORM_ONLY
from .reward import RewardConfig
from .task import TaskSpec, load_local_task

logger = logging.getLogger(__name__)


def initial_messages(episode: Episode, renderer: Renderer) -> list[Message]:
    specs = [t.to_spec() for t in episode.tools.all()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=specs, system_prompt=SYSTEM_PROMPT)
    instruction = TASK_INSTRUCTION_TRANSFORM_ONLY if episode.stage1_preloaded else TASK_INSTRUCTION
    return prefix + [{"role": "user", "content": instruction}]


def elt_rollout_config(max_turns: int, max_trajectory_tokens: int, max_tool_calls: int) -> RolloutConfig:
    """``agentic()`` preset with ELT-sized budgets: sequential tools (they share a
    filesystem and a warehouse), bounded parse-error retries, limit-stopped rollouts
    clamped to <= 0 so running out the clock is never rewarded."""
    return RolloutConfig(
        limits=RolloutLimits(max_turns=max_turns, max_trajectory_tokens=max_trajectory_tokens, max_tool_calls=max_tool_calls),
        parse_errors=ParseErrorPolicy(max_consecutive=2),
        termination=TerminationRewardPolicy(zero_reward_on_limit=True, skip_grading_on_timeout=True, grader_timeout_seconds=600),
        tool_execution="sequential",
    )


class ELTEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, task: TaskSpec, destination: Destination, rollouts_root: Path, group_size: int,
                 model_name: str, renderer_name: str | None, episode_cfg: EpisodeConfig,
                 max_turns: int = 40, max_trajectory_tokens: int = 28 * 1024, max_tool_calls: int = 60,
                 max_generation_tokens: int | None = None, context_overflow_reward: float = -0.1):
        self.task = task
        self.destination = destination
        self.rollouts_root = rollouts_root
        self.group_size = group_size
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.episode_cfg = episode_cfg
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_tool_calls = max_tool_calls
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self._episodes: list[Episode] = []

    async def make_envs(self) -> Sequence[Env]:
        base_tokenizer = tokenizer_utils.get_tokenizer(self.model_name)  # cached, shared object
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        cfg = elt_rollout_config(self.max_turns, self.max_trajectory_tokens, self.max_tool_calls)
        envs: list[Env] = []
        self._episodes = []
        for _ in range(self.group_size):
            # One tokenizer per env: rollouts tokenize concurrently in worker threads and a
            # shared HuggingFace fast tokenizer raises "Already borrowed" under that load.
            renderer = get_renderer(renderer_name, copy.deepcopy(base_tokenizer))
            ep = await self._create_episode()
            self._episodes.append(ep)
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=ep.tools.all(),
                    initial_messages=initial_messages(ep, renderer),
                    reward_fn=ep.reward_fn,
                    rollout_config=cfg,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return envs

    async def _create_episode(self) -> Episode:
        if self.task.mode == "local":
            return await create_local_episode(self.task, self.destination, self.rollouts_root, self.episode_cfg)
        from .official import create_official_episode  # imported lazily: needs warehouse + Airbyte deps

        return await create_official_episode(self.task, self.destination, self.rollouts_root, self.episode_cfg)

    async def cleanup(self) -> None:
        for ep in self._episodes:
            try:
                await ep.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.warning("episode cleanup failed: %s", e)
        self._episodes = []

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.mode, self.task.name]


class ELTDataset(RLDataset):
    """Cycles through tasks so every batch sees a rotating subset (no task starves)."""

    def __init__(self, builders: list[ELTEnvGroupBuilder], groups_per_batch: int, num_batches: int):
        self.builders = builders
        self.groups_per_batch = groups_per_batch
        self.num_batches = num_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n = len(self.builders)
        start = (index * self.groups_per_batch) % n
        return [self.builders[(start + i) % n] for i in range(self.groups_per_batch)]

    def __len__(self) -> int:
        return self.num_batches


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    task_dirs: list[str]
    """Local fixture directories (mode=local) or ELT-Bench task names (mode=eltbench)."""
    mode: str = "local"
    elt_bench_root: str | None = None
    destination: str = "duckdb"
    warehouse_root: str = "runs/warehouse"
    rollouts_root: str = "runs/rollouts"
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None
    group_size: int = 4
    groups_per_batch: int = 1
    num_batches: int = 1000
    max_turns: int = 40
    max_trajectory_tokens: int = 28 * 1024
    max_tool_calls: int = 60
    max_generation_tokens: int | None = None
    reward_mode: str = "staged"
    el_weight: float = 0.2
    column_credit: float = 0.5
    command_timeout: int = 600
    max_output_chars: int = 8 * 1024
    require_provenance: bool = True
    start_stage: str = "full"

    def _tasks(self) -> list[TaskSpec]:
        if self.mode == "local":
            return [load_local_task(d) for d in self.task_dirs]
        from .task import load_eltbench_task

        assert self.elt_bench_root, "elt_bench_root is required for mode=eltbench"
        return [load_eltbench_task(self.elt_bench_root, t, destination=self.destination) for t in self.task_dirs]

    def _destination(self) -> Destination:
        if self.destination == "duckdb":
            return DuckDBDestination(self.warehouse_root)
        if self.destination == "snowflake":
            from .destinations.snowflake_dest import SnowflakeDestination

            return SnowflakeDestination.from_env()
        raise ValueError(f"unknown destination {self.destination}")

    def _builders(self, group_size: int) -> list[ELTEnvGroupBuilder]:
        dest = self._destination()
        cfg = EpisodeConfig(
            reward=RewardConfig(mode=self.reward_mode, el_weight=self.el_weight, column_credit=self.column_credit),
            require_provenance=self.require_provenance,
            command_timeout=self.command_timeout,
            max_output_chars=self.max_output_chars,
            start_stage=self.start_stage,
        )
        return [
            ELTEnvGroupBuilder(task, dest, Path(self.rollouts_root).expanduser().resolve(), group_size, self.model_name, self.renderer_name, cfg,
                               self.max_turns, self.max_trajectory_tokens, self.max_tool_calls, self.max_generation_tokens)
            for task in self._tasks()
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train = ELTDataset(self._builders(self.group_size), self.groups_per_batch, self.num_batches)
        evaluation = ELTDataset(self._builders(1), max(1, min(self.groups_per_batch, len(self.task_dirs))), 1)
        return train, evaluation
