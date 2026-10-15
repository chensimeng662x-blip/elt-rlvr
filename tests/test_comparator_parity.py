"""The vendored comparator must agree with upstream ELT-Bench. Runs only when
ELT_BENCH_ROOT points at a checkout; otherwise the vendored semantics are still
exercised on hand-built cases."""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

from elt_rlvr import eltbench_compat as ours

CASES = [
    # (gold, candidate, keys, expected match)
    (pd.DataFrame({"k": ["1", "2"], "v": ["0E-9", "29324.840000000"]}), pd.DataFrame({"k": ["2", "1"], "V": ["29324.84", "0.0"]}), ["k"], True),
    (pd.DataFrame({"k": ["1"], "v": ["Abc "]}), pd.DataFrame({"k": ["1"], "v": ["abc"]}), ["k"], True),
    (pd.DataFrame({"k": ["1"], "v": [None]}), pd.DataFrame({"k": ["1"], "v": ["5"]}), ["k"], False),
    (pd.DataFrame({"k": ["1"], "v": ["1.0"]}), pd.DataFrame({"k": ["1"], "v": ["1.02"]}), ["k"], False),  # outside rtol 1e-2
    (pd.DataFrame({"k": ["1"], "v": ["100"]}), pd.DataFrame({"k": ["1"], "v": ["100.9"]}), ["k"], True),   # inside rtol
    (pd.DataFrame({"k": ["1"], "v": ["x"]}), pd.DataFrame({"k": ["1"], "extra": ["y"]}), ["k"], False),  # missed column
    (pd.DataFrame({"k": ["1"], "v": ["x"]}), pd.DataFrame({"k": ["1"], "v": ["x"], "extra": ["y"]}), ["k"], True),  # extra ok
]


@pytest.mark.parametrize("gt,cand,keys,expected", CASES)
def test_vendored_semantics(gt, cand, keys, expected):
    assert ours.compare_as_csv_strings(gt, cand, keys)["match"] is expected


def _load_upstream():
    root = os.environ.get("ELT_BENCH_ROOT")
    if not root:
        return None
    path = Path(root) / "evaluation" / "eva_stage2.py"
    if not path.exists():
        return None
    cwd = os.getcwd()
    os.chdir(path.parent)  # upstream loads sort_key.json relative to cwd at import
    try:
        spec = importlib.util.spec_from_file_location("eva_stage2_upstream", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


@pytest.mark.parametrize("gt,cand,keys,expected", CASES)
def test_parity_with_upstream(gt, cand, keys, expected):
    up = _load_upstream()
    if up is None:
        pytest.skip("ELT_BENCH_ROOT not set")
    g = up.sort_by_keys(gt.astype(str).where(gt.notna(), None), keys)
    c = up.sort_by_keys(cand.astype(str).where(cand.notna(), None), keys)
    theirs = up.check_corretness(g, c)["match"]
    mine = ours.check_corretness(ours.sort_by_keys(gt.astype(str).where(gt.notna(), None), keys),
                                 ours.sort_by_keys(cand.astype(str).where(cand.notna(), None), keys))["match"]
    assert theirs == mine == expected
