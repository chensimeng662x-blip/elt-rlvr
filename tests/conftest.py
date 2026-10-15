import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "mini_shop"


@pytest.fixture
def mini_shop():
    from elt_rlvr.task import load_local_task

    return load_local_task(FIXTURE)


@pytest.fixture
def duckdb_dest(tmp_path):
    from elt_rlvr.destinations import DuckDBDestination

    return DuckDBDestination(tmp_path / "warehouse")
