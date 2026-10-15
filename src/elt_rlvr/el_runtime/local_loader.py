"""Credential-free EL runtime.

The agent writes ``elt/el_config.yaml`` following ``documentation/local_el.md``
(field names deliberately differ from ``config.yaml``, as in ELT-Bench where
Airbyte's Terraform fields do not match the task config). ``run()`` validates
the file, loads every stream into the rollout's warehouse namespace with the
same ``_airbyte_*`` metadata columns Airbyte adds, and records provenance.

Supported connectors: ``file`` (csv, jsonl, parquet) and ``sqlite``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .base import ELReport, ELRuntime, StreamProvenance, fingerprint_table

SYNC_MODES = {"full_refresh_overwrite", "full_refresh_append"}

LOCAL_EL_DOC = """# Local EL connector (credential-free runtime)

The extraction and loading step is executed by the `run_el` tool. It reads
`/workspace/elt/el_config.yaml` and loads every declared stream into the
destination schema. Each loaded table gets the Airbyte metadata columns
`_airbyte_raw_id`, `_airbyte_extracted_at`, `_airbyte_meta`,
`_airbyte_generation_id` in addition to the source columns.

## el_config.yaml

```yaml
version: 1
destination:
  connector: duckdb
  database_path: <destination.database from config.yaml>
  schema: <destination.schema from config.yaml>
sources:
  - connector: file            # flat files: csv, jsonl or parquet
    file_format: csv           # csv | jsonl | parquet
    url: file:///absolute/path/to/file.csv
    stream: customers          # table name created in the destination
    sync_mode: full_refresh_overwrite
  - connector: sqlite          # a SQLite database file
    database_path: /absolute/path/to/db.sqlite
    streams:
      - name: products         # one entry per table to replicate
        sync_mode: full_refresh_overwrite
```

Rules:
- `sync_mode` is `full_refresh_overwrite` (replace the table) or
  `full_refresh_append` (append; running twice duplicates rows).
- Every stream in `config.yaml` must be declared exactly once.
- Values are the realistic values from `config.yaml`; do not invent paths.
- `run_el` returns one job per stream with `status` and `records_synced`.
  A failed job explains the error; fix the configuration and run again.
"""


class LocalELRuntime(ELRuntime):
    name = "local"

    def __init__(self, handle, workdir, config_relpath: str = "elt/el_config.yaml", allowed_roots=None):
        super().__init__(handle, workdir)
        self.config_relpath = config_relpath
        self.allowed_roots = [Path(p).resolve() for p in (allowed_roots or [])]
        self._generation = 0

    def agent_documentation(self) -> dict[str, str]:
        return {"local_el.md": LOCAL_EL_DOC}

    # -- helpers -------------------------------------------------------------
    def _check_path(self, raw: str) -> Path:
        if raw.startswith("file://"):
            raw = raw[len("file://"):]
        p = Path(raw).expanduser().resolve()
        if self.allowed_roots and not any(p == r or r in p.parents for r in self.allowed_roots):
            raise ValueError(f"path not allowed: {raw}")
        if not p.exists():
            raise FileNotFoundError(f"source not found: {raw}")
        return p

    @staticmethod
    def _read_file(path: Path, fmt: str) -> pd.DataFrame:
        fmt = fmt.lower()
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "jsonl":
            return pd.read_json(path, lines=True)
        if fmt == "parquet":
            return pd.read_parquet(path)
        raise ValueError(f"unsupported file_format: {fmt}")

    @staticmethod
    def _read_sqlite(path: Path, table: str) -> pd.DataFrame:
        with sqlite3.connect(str(path)) as c:
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if table not in names:
                raise ValueError(f"sqlite table not found: {table} (available: {sorted(names)})")
            return pd.read_sql_query(f'SELECT * FROM "{table}"', c)

    def _with_airbyte_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        df = df.copy()
        df["_airbyte_raw_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df["_airbyte_extracted_at"] = now
        df["_airbyte_meta"] = json.dumps({"changes": [], "sync_id": self._generation})
        df["_airbyte_generation_id"] = self._generation
        return df

    # -- main ----------------------------------------------------------------
    async def run(self) -> ELReport:
        import duckdb

        cfg_path = Path(self.workdir) / self.config_relpath
        if not cfg_path.exists():
            return ELReport(False, {"error": f"{self.config_relpath} not found in workspace"})
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError as e:
            return ELReport(False, {"error": f"invalid YAML: {e}"})

        errors: list[str] = []
        dest = cfg.get("destination") or {}
        if dest.get("connector") != "duckdb":
            errors.append("destination.connector must be 'duckdb'")
        schema = dest.get("schema")
        if schema != self.handle.schema:
            errors.append(f"destination.schema must be '{self.handle.schema}' (got {schema!r})")
        if str(dest.get("database_path", "")) != str(self.handle.extra["path"]):
            errors.append("destination.database_path does not match config.yaml")

        plan: list[tuple[str, str, object]] = []  # (stream, sync_mode, loader)
        sources = cfg.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append("sources must be a non-empty list")
            sources = []
        for i, src in enumerate(sources):
            try:
                conn = src.get("connector")
                if conn == "file":
                    path = self._check_path(str(src.get("url", "")))
                    fmt = src.get("file_format")
                    stream = src.get("stream")
                    mode = src.get("sync_mode", "")
                    if not stream:
                        raise ValueError("file source needs `stream`")
                    if mode not in SYNC_MODES:
                        raise ValueError(f"sync_mode must be one of {sorted(SYNC_MODES)}")
                    plan.append((stream, mode, (lambda p=path, f=fmt: self._read_file(p, f))))
                elif conn == "sqlite":
                    path = self._check_path(str(src.get("database_path", "")))
                    streams = src.get("streams") or []
                    if not streams:
                        raise ValueError("sqlite source needs `streams`")
                    for s in streams:
                        name = s.get("name") if isinstance(s, dict) else None
                        mode = s.get("sync_mode", "") if isinstance(s, dict) else ""
                        if not name:
                            raise ValueError("each sqlite stream needs `name`")
                        if mode not in SYNC_MODES:
                            raise ValueError(f"stream {name}: sync_mode must be one of {sorted(SYNC_MODES)}")
                        plan.append((name, mode, (lambda p=path, t=name: self._read_sqlite(p, t))))
                else:
                    raise ValueError(f"unknown connector {conn!r} (file | sqlite)")
            except Exception as e:  # noqa: BLE001 - every config error goes back to the model
                errors.append(f"sources[{i}]: {e}")

        seen = set()
        for stream, _, _ in plan:
            if stream in seen:
                errors.append(f"stream '{stream}' declared more than once")
            seen.add(stream)

        if errors:
            return ELReport(False, {"status": "config_error", "errors": errors})

        self._generation += 1
        jobs = []
        ok = True
        db = duckdb.connect(self.handle.extra["path"])
        try:
            db.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            for stream, mode, loader in plan:
                job_id = f"local-{self._generation}-{uuid.uuid4().hex[:6]}"
                try:
                    df = self._with_airbyte_meta(loader())
                    db.register("_incoming", df)
                    fq = f'"{schema}"."{stream}"'
                    if mode == "full_refresh_overwrite":
                        db.execute(f"CREATE OR REPLACE TABLE {fq} AS SELECT * FROM _incoming")
                    else:
                        exists = db.execute(
                            "SELECT count(*) FROM information_schema.tables WHERE lower(table_schema)=? AND lower(table_name)=?",
                            [schema.lower(), stream.lower()],
                        ).fetchone()[0]
                        if exists:
                            db.execute(f"INSERT INTO {fq} SELECT * FROM _incoming")
                        else:
                            db.execute(f"CREATE TABLE {fq} AS SELECT * FROM _incoming")
                    db.unregister("_incoming")
                    jobs.append({"job_id": job_id, "stream": stream, "status": "succeeded", "records_synced": int(len(df))})
                except Exception as e:  # noqa: BLE001
                    ok = False
                    jobs.append({"job_id": job_id, "stream": stream, "status": "failed", "error": str(e)})
        finally:
            db.close()

        # Provenance is fingerprinted from the warehouse, not from the DataFrame,
        # so the grader's re-fingerprint uses an identical code path.
        with self.handle.connector(read_only=True) as c:
            for j in jobs:
                if j["status"] != "succeeded":
                    continue
                rows, fp = fingerprint_table(c, self.handle.database, schema, j["stream"])
                self.provenance[j["stream"].lower()] = StreamProvenance(
                    table=j["stream"], rows=rows, fingerprint=fp, job_id=j["job_id"]
                )
        return ELReport(ok, {"status": "succeeded" if ok else "partial_failure", "jobs": jobs}, dict(self.provenance))
