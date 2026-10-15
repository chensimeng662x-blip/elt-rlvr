"""Run one episode on an official ELT-Bench task with a scripted or sampled policy.

    python scripts/run_official.py --elt-bench-root ../ELT-Bench --task address --print-grade
    python scripts/run_official.py --elt-bench-root ../ELT-Bench --task address --model Qwen/Qwen3-8B

With --model, the policy is sampled from Tinker's base model (no training); the
rollout goes through the same EnvGroupBuilder the trainer uses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from elt_rlvr.destinations.snowflake_dest import SnowflakeDestination
from elt_rlvr.episode import EpisodeConfig
from elt_rlvr.official import create_official_episode
from elt_rlvr.task import load_eltbench_task


async def scripted(args) -> None:
    task = load_eltbench_task(args.elt_bench_root, args.task, destination=args.destination)
    dest = SnowflakeDestination.from_env()
    ep = await create_official_episode(task, dest, args.rollouts_root, EpisodeConfig())
    try:
        # Minimal smoke path: provision, run_el on whatever is in elt/, grade.
        print("workspace:", ep.workdir)
        print("namespace:", ep.handle.extra)
        if args.run_el:
            r = await ep.el_runtime.run()
            print(json.dumps(r.public, indent=2)[:4000])
        report = ep.grade()
        print(json.dumps(report.summary(), indent=2))
    finally:
        if not args.keep:
            await ep.cleanup()


async def sampled(args) -> None:
    import tinker
    from tinker_cookbook import model_info, renderers, tokenizer_utils
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_single_rollout

    from elt_rlvr.env import ELTEnvGroupBuilder

    task = load_eltbench_task(args.elt_bench_root, args.task, destination=args.destination)
    dest = SnowflakeDestination.from_env()
    gb = ELTEnvGroupBuilder(task, dest, Path(args.rollouts_root), 1, args.model, None, EpisodeConfig(), max_turns=args.max_turns)
    service = tinker.ServiceClient()
    sampling = service.create_sampling_client(base_model=args.model)
    completer = TinkerTokenCompleter(sampling, max_tokens=4096)
    envs = await gb.make_envs()
    try:
        traj = await do_single_rollout(completer, envs[0])
        print("reward:", sum(t.reward for t in traj.transitions), "turns:", len(traj.transitions))
        print(json.dumps(gb._episodes[0].last_report.summary() if gb._episodes[0].last_report else {}, indent=2))
    finally:
        await gb.cleanup()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--elt-bench-root", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--destination", default="snowflake")
    p.add_argument("--rollouts-root", default="runs/rollouts")
    p.add_argument("--model", default=None, help="sample a policy from this Tinker base model")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--run-el", action="store_true", help="scripted mode: call run_el on the shipped elt/ dir")
    p.add_argument("--print-grade", action="store_true")
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()
    asyncio.run(sampled(args) if args.model else scripted(args))


if __name__ == "__main__":
    main()
