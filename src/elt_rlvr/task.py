"""Task specifications.

A task is split into two halves that must never be mixed:

- ``agent_dir``: what the policy may see and edit (``config.yaml``,
  ``data_model.yaml``, ``schemas/``, ``documentation/``, ``elt/``).
- ``grader``: what only the host-side grader reads (expected raw-table row
  counts, ground-truth CSVs or SQL, sort keys).

Two loaders exist: ``load_local_task`` for the credential-free fixtures shipped
in this repo, and ``load_eltbench_task`` for an official ELT-Bench checkout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GraderSpec:
    """Grader-only half of a task."""

    expected_row_counts: dict[str, int]
    """Raw table name -> expected row count (ELT-Bench ``table.json`` entry)."""
    sort_keys: dict[str, list[str]]
    """Data model name -> unique key columns (ELT-Bench ``sort_key.json`` entry)."""
    gt_csv_dir: Path | None
    """Directory with ``<model>.csv`` ground truth files."""
    gt_sql_dir: Path | None = None
    """Directory with ``<model>.sql`` queries that fetch the candidate table.
    Falls back to ``SELECT *`` when a model has no query (upstream behaviour)."""

    @property
    def model_names(self) -> list[str]:
        if self.gt_csv_dir is not None and self.gt_csv_dir.is_dir():
            return sorted(p.stem for p in self.gt_csv_dir.glob("*.csv"))
        return sorted(self.sort_keys)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    """Logical database / schema name (``address``, ``mini_shop``...)."""
    agent_dir: Path
    """Directory copied into every rollout workspace."""
    grader: GraderSpec
    mode: str
    """``local`` (credential-free runtime) or ``eltbench`` (Airbyte + warehouse)."""
    sources_dir: Path | None = None
    """Local mode only: directory with the raw source files."""
    extra: dict = field(default_factory=dict)

    @property
    def data_model_names(self) -> list[str]:
        dm = yaml.safe_load((self.agent_dir / "data_model.yaml").read_text())
        return [m["name"] for m in dm.get("models", [])]

    @property
    def raw_table_names(self) -> list[str]:
        return sorted(self.grader.expected_row_counts)


def load_local_task(fixture_dir: str | Path) -> TaskSpec:
    """Load a fixture task laid out as::

        <fixture>/agent/      config.yaml, data_model.yaml, schemas/, documentation/, elt/
        <fixture>/sources/    raw files the local EL runtime reads
        <fixture>/grader/     table.json, sort_key.json, gt/<model>.csv
    """
    fixture_dir = Path(fixture_dir).resolve()
    grader_dir = fixture_dir / "grader"
    table = json.loads((grader_dir / "table.json").read_text())
    sort_keys = json.loads((grader_dir / "sort_key.json").read_text())
    name = fixture_dir.name
    return TaskSpec(
        name=name,
        agent_dir=fixture_dir / "agent",
        grader=GraderSpec(
            expected_row_counts={k: int(v) for k, v in table.items()},
            sort_keys={k: list(v) for k, v in sort_keys.items()},
            gt_csv_dir=grader_dir / "gt",
            gt_sql_dir=grader_dir / "sql" if (grader_dir / "sql").is_dir() else None,
        ),
        mode="local",
        sources_dir=fixture_dir / "sources",
    )


def load_eltbench_task(
    elt_bench_root: str | Path,
    task_name: str,
    destination: str = "snowflake",
    inputs_dir: str | Path | None = None,
    gt_dir: str | Path | None = None,
) -> TaskSpec:
    """Load one official ELT-Bench task.

    ``inputs_dir`` is the folder produced by ``setup/write_config.py`` (default
    ``<root>/inputs`` for Snowflake). ``gt_dir`` holds the ground-truth CSVs
    (``evaluation/agent_results/gt_snowflake/<task>/`` after ``elt_setup.sh``).
    """
    root = Path(elt_bench_root).resolve()
    inputs = Path(inputs_dir) if inputs_dir else root / {
        "snowflake": "inputs",
        "databricks": "inputs_databricks",
        "redshift": "inputs_redshift",
    }[destination]
    agent_dir = inputs / task_name
    if not agent_dir.is_dir():
        raise FileNotFoundError(
            f"{agent_dir} not found. Run `python setup/write_config.py --destination {destination}` "
            "in the ELT-Bench checkout first."
        )
    table = json.loads((root / "evaluation/table.json").read_text())[task_name]
    sort_keys = json.loads((root / "evaluation/sort_key.json").read_text()).get(task_name, {})
    gt = Path(gt_dir) if gt_dir else root / "evaluation/agent_results" / f"gt_{destination}" / task_name
    return TaskSpec(
        name=task_name,
        agent_dir=agent_dir,
        grader=GraderSpec(
            expected_row_counts={k: int(v) for k, v in table.items()},
            sort_keys={k: list(v) for k, v in sort_keys.items()},
            gt_csv_dir=gt if gt.is_dir() else None,
            gt_sql_dir=root / "evaluation/sql" / task_name,
        ),
        mode="eltbench",
        extra={"destination": destination, "elt_bench_root": str(root)},
    )
