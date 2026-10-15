"""Command-line helpers.

    elt-rlvr demo-local [--policy oracle] [--fixture fixtures/mini_shop]
    elt-rlvr audit                      # every scripted policy, reward matrix
    elt-rlvr grade-only --fixture ...   # grade an existing namespace (debug)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from .destinations import DuckDBDestination
from .episode import EpisodeConfig, create_local_episode
from .policies import POLICIES
from .reward import RewardConfig
from .scripted import run_scripted
from .task import load_local_task


async def _episode(fixture: str, root: Path, reward_mode: str, keep: bool):
    task = load_local_task(fixture)
    dest = DuckDBDestination(root / "warehouse")
    cfg = EpisodeConfig(reward=RewardConfig(mode=reward_mode))
    return await create_local_episode(task, dest, root / "rollouts", cfg)


async def demo_local(args) -> int:
    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="elt-rlvr-"))
    ep = await _episode(args.fixture, root, args.reward_mode, args.keep)
    try:
        r = await run_scripted(ep, args.policy, verbose=args.verbose)
    finally:
        if not args.keep:
            await ep.cleanup()
    print(json.dumps({"policy": r.policy, "reward": r.reward, "turns": r.turns, "grade": r.report}, indent=2))
    if args.keep:
        print(f"workspace kept at {ep.workdir}")
    ok = (r.reward == 1.0) if args.policy == "oracle" else True
    return 0 if ok else 1


async def audit(args) -> int:
    root = Path(tempfile.mkdtemp(prefix="elt-rlvr-audit-"))
    rows = []
    for mode in ("binary", "model_fraction", "staged"):
        for name in POLICIES:
            ep = await _episode(args.fixture, root, mode, False)
            try:
                r = await run_scripted(ep, name)
            finally:
                await ep.cleanup()
            rows.append((mode, name, r.reward, r.report["el_fraction"], r.report["models_correct"], r.report["mean_column_fraction"]))
    print(f"{'reward_mode':<15}{'policy':<18}{'reward':>8}{'el_frac':>9}{'models_ok':>11}{'col_frac':>10}")
    for m, n, rw, ef, mc, cf in rows:
        print(f"{m:<15}{n:<18}{rw:>8.3f}{ef:>9.2f}{mc:>11d}{cf:>10.2f}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="elt-rlvr")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo-local", help="run a scripted policy through a full local episode")
    d.add_argument("--fixture", default="fixtures/mini_shop")
    d.add_argument("--policy", default="oracle", choices=sorted(POLICIES))
    d.add_argument("--reward-mode", default="staged")
    d.add_argument("--root", default=None)
    d.add_argument("--keep", action="store_true")
    d.add_argument("--verbose", action="store_true")
    a = sub.add_parser("audit", help="reward matrix over all scripted policies and reward modes")
    a.add_argument("--fixture", default="fixtures/mini_shop")
    args = p.parse_args()
    fn = {"demo-local": demo_local, "audit": audit}[args.cmd]
    raise SystemExit(asyncio.run(fn(args)))
