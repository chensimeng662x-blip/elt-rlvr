"""Transform-only curriculum start: Stage 1 is preloaded through the sanctioned EL
runtime (so provenance exists), the agent only writes models, and the grader treats
the preloaded tables exactly like agent-loaded ones."""

import pytest

from elt_rlvr.episode import EpisodeConfig, create_local_episode
from elt_rlvr.scripted import run_scripted


async def test_transform_only_preload_then_models(mini_shop, duckdb_dest, tmp_path):
    ep = await create_local_episode(mini_shop, duckdb_dest, tmp_path / "r", EpisodeConfig(start_stage="transform"))
    try:
        assert ep.stage1_preloaded
        assert set(ep.el_runtime.provenance) == {"customers", "orders", "products"}
        assert not (ep.workdir / "elt" / "el_config.yaml").exists()  # no answer key left in the workspace
        # an agent that only writes correct models (no run_el) gets full credit
        from elt_rlvr.policies import POLICIES, _dbt_steps, CORRECT_MODELS
        import yaml
        POLICIES["_models_only"] = lambda cfg: list(_dbt_steps(cfg, CORRECT_MODELS)) + [("submit", {})]
        r = await run_scripted(ep, "_models_only")
        assert r.report["el_complete"]
        assert r.report["models_correct"] == 2
        assert r.reward == pytest.approx(1.0)
    finally:
        POLICIES.pop("_models_only", None)
        await ep.cleanup()
