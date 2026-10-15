"""Run a scripted policy through the real ``AgentToolMessageEnv`` (no renderer, no
tokenizer, no Tinker). This exercises the exact tool dispatch, stop semantics and
grading path the RL loop uses, minus sampling."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

import yaml
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from .episode import Episode
from .policies import POLICIES


@dataclass
class ScriptedResult:
    policy: str
    reward: float
    metrics: dict[str, float]
    turns: int
    transcript: list[dict] = field(default_factory=list)
    report: dict | None = None


def _assistant(tool: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": f"Calling {tool}.",
        "tool_calls": [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", function=ToolCall.FunctionBody(name=tool, arguments=json.dumps(args)))],
    }


async def run_scripted(episode: Episode, policy: str, max_turns: int = 40, verbose: bool = False) -> ScriptedResult:
    config = yaml.safe_load((episode.workdir / "config.yaml").read_text())
    steps = list(POLICIES[policy](config))
    env = AgentToolMessageEnv(
        tools=episode.tools.all(),
        initial_messages=[{"role": "system", "content": "scripted"}, {"role": "user", "content": episode.task.name}],
        max_turns=max_turns,
        reward_fn=episode.reward_fn,
        tool_execution="sequential",
    )
    await env.initial_observation()
    transcript: list[dict] = []
    total = 0.0
    metrics: dict[str, float] = {}
    turns = 0
    done = False
    for tool, args in steps:
        turns += 1
        result = await env.step(_assistant(tool, args))
        total += result.reward
        metrics.update(result.metrics)
        last = result.next_messages[-1]
        entry = {"tool": tool, "args": args, "result": str(last.get("content"))[:2000]}
        transcript.append(entry)
        if verbose:
            print(f"[{turns}] {tool}({json.dumps(args)[:120]}) -> {entry['result'][:400]}")
        if result.episode_done:
            done = True
            break
    if not done:
        # Policy ran out of steps without submitting: a final assistant message with no
        # tool calls ends the episode the same way a model that just stops would.
        turns += 1
        result = await env.step({"role": "assistant", "content": "Done."})
        total += result.reward
        metrics.update(result.metrics)
    report = episode.last_report.summary() if episode.last_report else None
    return ScriptedResult(policy, total, metrics, turns, transcript, report)
