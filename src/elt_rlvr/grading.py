"""Host-side grading of a rollout's warehouse namespace.

Stage 1 (extract & load) and Stage 2 (transform) follow ELT-Bench's evaluators
exactly, with one addition used only for RL: raw-table *provenance*. The
grader never looks at the transcript, the workspace, or anything the policy
can write to except the warehouse itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .destinations.base import DestinationHandle
from .el_runtime.base import AIRBYTE_META_COLUMNS, StreamProvenance, fingerprint_table
from .eltbench_compat import compare_as_csv_strings
from .task import TaskSpec


@dataclass
class RawTableGrade:
    table: str
    expected_rows: int
    found: bool
    rows: int | None
    count_ok: bool
    provenance_ok: bool | None
    """None when provenance was not required/available; False when the table exists
    but its content fingerprint differs from what the sanctioned EL path loaded."""

    @property
    def ok(self) -> bool:
        return self.found and self.count_ok and (self.provenance_ok is not False)


@dataclass
class ModelGrade:
    model: str
    found: bool
    match: bool
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def total_columns(self) -> int:
        return len(self.matched) + len(self.unmatched) + len(self.missed)

    @property
    def column_fraction(self) -> float:
        return len(self.matched) / self.total_columns if self.total_columns else 0.0


@dataclass
class GradeReport:
    task: str
    raw_tables: list[RawTableGrade]
    models: list[ModelGrade]
    schema_found: bool
    submitted: bool

    @property
    def el_fraction(self) -> float:
        return sum(t.ok for t in self.raw_tables) / len(self.raw_tables) if self.raw_tables else 0.0

    @property
    def el_complete(self) -> bool:
        return bool(self.raw_tables) and all(t.ok for t in self.raw_tables)

    @property
    def models_correct(self) -> int:
        return sum(m.match for m in self.models)

    @property
    def model_fraction(self) -> float:
        return self.models_correct / len(self.models) if self.models else 0.0

    @property
    def mean_column_fraction(self) -> float:
        return sum(m.column_fraction for m in self.models) / len(self.models) if self.models else 0.0

    def summary(self) -> dict:
        return {
            "task": self.task,
            "submitted": self.submitted,
            "schema_found": self.schema_found,
            "el_fraction": round(self.el_fraction, 4),
            "el_complete": self.el_complete,
            "models_correct": self.models_correct,
            "models_total": len(self.models),
            "mean_column_fraction": round(self.mean_column_fraction, 4),
            "raw_tables": [asdict(t) | {"ok": t.ok} for t in self.raw_tables],
            "models": [asdict(m) | {"column_fraction": round(m.column_fraction, 4)} for m in self.models],
        }


def grade(task: TaskSpec, handle: DestinationHandle, provenance: dict[str, StreamProvenance] | None,
          submitted: bool, require_provenance: bool = True) -> GradeReport:
    g = task.grader
    with handle.connector(read_only=True) as c:
        schema_found = c.verify_schema(handle.database, handle.schema)
        existing = {c._norm(t): t for t in c.list_tables(handle.database, handle.schema)} if schema_found else {}

        raw: list[RawTableGrade] = []
        for table, expected in sorted(g.expected_row_counts.items()):
            actual = existing.get(c._norm(table))
            if actual is None:
                raw.append(RawTableGrade(table, expected, False, None, False, None))
                continue
            rows = c.table_size(handle.database, handle.schema, actual)
            prov_ok: bool | None = None
            if require_provenance:
                p = (provenance or {}).get(table.lower())
                if p is None:
                    prov_ok = False
                else:
                    cols = {x.lower() for x in c.list_columns(handle.database, handle.schema, actual)}
                    has_meta = {"_airbyte_raw_id", "_airbyte_extracted_at"} <= cols
                    _, fp = fingerprint_table(c, handle.database, handle.schema, actual)
                    prov_ok = has_meta and fp == p.fingerprint
            raw.append(RawTableGrade(table, expected, True, rows, rows == expected, prov_ok))

        models: list[ModelGrade] = []
        for name in g.model_names:
            gt_path = g.gt_csv_dir / f"{name}.csv" if g.gt_csv_dir else None
            if gt_path is None or not gt_path.exists():
                models.append(ModelGrade(name, False, False, error="ground truth missing"))
                continue
            df_gt = pd.read_csv(gt_path, dtype=str, keep_default_na=True)
            actual = existing.get(c._norm(name))
            if actual is None:
                models.append(ModelGrade(name, False, False, missed=list(df_gt.columns)))
                continue
            try:
                sql_path = g.gt_sql_dir / f"{name}.sql" if g.gt_sql_dir else None
                if sql_path is not None and sql_path.exists():
                    df = c.query_evaluation_sql(sql_path.read_text(), handle.database)
                else:
                    df = c.fetch_table(handle.database, handle.schema, actual)
                res = compare_as_csv_strings(df_gt, df, g.sort_keys.get(name, []))
                models.append(ModelGrade(name, True, bool(res["match"]), res["matched"], res["unmatched"], res["missed"]))
            except Exception as e:  # noqa: BLE001
                models.append(ModelGrade(name, True, False, missed=list(df_gt.columns), error=str(e)))

    return GradeReport(task.name, raw, models, schema_found, submitted)
