"""Vendored copy of the ELT-Bench stage-2 comparator.

Source: uiuc-kang-lab/ELT-Bench, branch ``elt_bench++``, ``evaluation/eva_stage2.py``
(CC BY-SA 4.0). Only the pure functions are copied, so grading does not need the
benchmark checkout or any warehouse driver. ``tests/test_comparator_parity.py``
checks this copy against the upstream file when ``ELT_BENCH_ROOT`` is set.

Semantics (unchanged from upstream):
- both sides are read as strings, then re-sorted by the table's unique key with
  every other column as a tie-breaker (``sort_by_keys``);
- a gold column matches when both sides coerce to numeric and are close
  (rtol=1e-2, atol=1e-9, NaN==NaN), or otherwise when trimmed, lower-cased
  strings are equal; NaN vs value never matches;
- extra columns in the candidate are allowed; every gold column must match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sort_series_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    n_non_null = col.notna().sum()
    if n_non_null > 0 and int(num.notna().sum()) == int(n_non_null):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    if df is None or df.empty:
        return df.reset_index(drop=True) if df is not None else df

    lower = {c.lower(): c for c in df.columns}
    resolved = []
    for k in keys:
        actual = lower.get(k.lower())
        if actual is not None and actual not in resolved:
            resolved.append(actual)
    ordered_cols = resolved + [c for c in df.columns if c not in resolved]
    if not ordered_cols:
        return df.reset_index(drop=True)

    aux = pd.DataFrame({c: _sort_series_key(df[c]) for c in ordered_cols})
    order = aux.sort_values(by=ordered_cols, kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.isna().sum() == s.isna().sum():
        return coerced
    return s


def _vectors_match(v1: pd.Series, v2: pd.Series, rtol: float = 1e-2, atol: float = 1e-9) -> bool:
    if len(v1) != len(v2):
        return False

    n1 = _to_numeric_if_possible(v1.reset_index(drop=True))
    n2 = _to_numeric_if_possible(v2.reset_index(drop=True))
    both_numeric = pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2)

    if both_numeric:
        a = n1.to_numpy(dtype=float)
        b = n2.to_numpy(dtype=float)
        nan_match = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        return bool(np.all(nan_match | close))

    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            return False
        if isinstance(a, str) and isinstance(b, str):
            if a.strip().lower() != b.strip().lower():
                return False
        else:
            if a != b:
                return False
    return True


def check_corretness(df_gt: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Upstream name kept (including the typo) so parity tests read naturally."""
    matched_cols, unmatched_cols, missed_cols = [], [], []
    df_cols_lower = {c.lower(): c for c in df.columns}

    for gold_col in df_gt.columns:
        actual_col = df_cols_lower.get(gold_col.lower())
        if actual_col is None:
            missed_cols.append(gold_col)
            continue
        if _vectors_match(df_gt[gold_col], df[actual_col]):
            matched_cols.append(gold_col)
        else:
            unmatched_cols.append(gold_col)

    return {
        "status": "ok",
        "match": not unmatched_cols and not missed_cols,
        "matched": matched_cols,
        "unmatched": unmatched_cols,
        "missed": missed_cols,
    }


def compare_as_csv_strings(df_gt: pd.DataFrame, df: pd.DataFrame, keys: list) -> dict:
    """Mirror the upstream driver: round-trip both frames through CSV text so
    dtype inference is identical to the official evaluator, then sort and compare."""
    import io

    def roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        buf.seek(0)
        return pd.read_csv(buf, dtype=str, keep_default_na=True)

    gt = sort_by_keys(roundtrip(df_gt), keys)
    cand = sort_by_keys(roundtrip(df), keys)
    res = check_corretness(gt, cand)
    res["sort_keys"] = list(keys)
    res["gt_rows"] = int(len(gt))
    res["candidate_rows"] = int(len(cand))
    return res
