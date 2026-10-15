"""Credential-free integration tests: full episodes through AgentToolMessageEnv,
graded by warehouse state, no network, no accounts, no Docker."""

import pytest

from elt_rlvr.episode import EpisodeConfig, create_local_episode
from elt_rlvr.reward import RewardConfig
from elt_rlvr.scripted import run_scripted


async def _run(mini_shop, duckdb_dest, tmp_path, policy, **reward_kwargs):
    cfg = EpisodeConfig(reward=RewardConfig(**reward_kwargs))
    ep = await create_local_episode(mini_shop, duckdb_dest, tmp_path / "rollouts", cfg)
    try:
        return await run_scripted(ep, policy)
    finally:
        await ep.cleanup()


async def test_oracle_scores_one(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "oracle")
    assert r.report["el_complete"], r.report
    assert r.report["models_correct"] == 2, r.report
    assert r.reward == pytest.approx(1.0)


async def test_binary_mode_matches_official_definition(mini_shop, duckdb_dest, tmp_path):
    assert (await _run(mini_shop, duckdb_dest, tmp_path, "oracle", mode="binary")).reward == 1.0
    assert (await _run(mini_shop, duckdb_dest, tmp_path, "wrong_models", mode="binary")).reward == 0.0


async def test_unsubmitted_episode_scores_zero(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "no_submit")
    assert r.report["models_correct"] == 2  # pipeline was right...
    assert r.reward == 0.0                  # ...but never submitted


async def test_el_only_gets_el_weight(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "el_only", el_weight=0.2)
    assert r.report["el_complete"]
    assert r.reward == pytest.approx(0.2)


async def test_wrong_models_get_partial_column_credit_only(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "wrong_models", el_weight=0.2, column_credit=0.5)
    assert r.report["models_correct"] == 0
    frac = r.report["mean_column_fraction"]
    assert 0 < frac < 1
    assert r.reward == pytest.approx(0.2 + 0.8 * 0.5 * frac)
    assert r.reward < 0.6


async def test_views_are_not_tables(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "views_not_tables")
    assert r.report["models_correct"] == 0
    assert all(not m["found"] for m in r.report["models"])


async def test_fake_load_fails_provenance(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "fake_load")
    raw = {t["table"]: t for t in r.report["raw_tables"]}
    assert all(t["count_ok"] for t in raw.values())          # counts look perfect...
    assert all(t["provenance_ok"] is False for t in raw.values())  # ...but nothing came through run_el
    assert not r.report["el_complete"]
    assert r.reward == 0.0


async def test_append_twice_doubles_counts(mini_shop, duckdb_dest, tmp_path):
    r = await _run(mini_shop, duckdb_dest, tmp_path, "append_twice")
    raw = {t["table"]: t for t in r.report["raw_tables"]}
    assert raw["orders"]["rows"] == 30 and not raw["orders"]["count_ok"]
    assert r.report["el_fraction"] == 0.0
    assert r.reward == 0.0
