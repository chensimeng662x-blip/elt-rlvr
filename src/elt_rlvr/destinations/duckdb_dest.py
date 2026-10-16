"""DuckDB destination: one database file per rollout, zero credentials.

Layout mirrors the Snowflake one the evaluators expect: the logical task
database is a schema inside the file, raw tables and dbt models both live in
it, and evaluation SQL written as ``schema.table`` resolves unchanged.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import duckdb

from .base import BaseConnector, Destination, DestinationHandle


class DuckDBConnector(BaseConnector):
    def _connect(self):
        return duckdb.connect(self.config["path"], read_only=self.config.get("read_only", False))

    def query(self, sql: str):
        # duckdb connections are not DB-API cursors with .description on the connection
        conn = self.conn()
        rel = conn.execute(sql)
        cols = [d[0] for d in rel.description] if rel.description else []
        rows = rel.fetchall()
        import pandas as pd

        return pd.DataFrame(rows, columns=cols)

    def _quote(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def _norm(self, ident: str) -> str:
        return ident.lower()

    def _schemata_sql(self, database: str) -> str:
        return "SELECT schema_name FROM information_schema.schemata"

    def _tables_sql(self, database: str, schema: str) -> str:
        return (
            "SELECT table_name FROM information_schema.tables "
            f"WHERE lower(table_schema) = '{schema.lower()}' AND table_type = 'BASE TABLE'"
        )

    def _fqn(self, database: str, schema: str, table: str) -> str:
        return f"{self._quote(schema)}.{self._quote(table)}"

    def prepare_evaluation_sql(self, sql: str, database: str) -> str:
        return self._rewrite_relations(sql, lambda s, t: f"{self._quote(s)}.{self._quote(t)}")


class DuckDBDestination(Destination):
    name = "duckdb"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task_name: str, rollout_id: str) -> Path:
        return self.root / f"{task_name}-{rollout_id}.duckdb"

    async def provision(self, task_name: str, rollout_id: str) -> DestinationHandle:
        path = self._path(task_name, rollout_id)
        if path.exists():
            path.unlink()
        conn = duckdb.connect(str(path))
        try:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{task_name}"')
        finally:
            conn.close()
        return DestinationHandle(
            destination=self,
            database=path.stem,
            schema=task_name,
            agent_config={
                "database": str(path),
                "schema": task_name,
                "dbt_profile": "elt",
                "dbt_target": "dev",
            },
            dbt_profile={
                "elt": {
                    "target": "dev",
                    "outputs": {
                        "dev": {"type": "duckdb", "path": str(path), "schema": task_name, "threads": 1}
                    },
                }
            },
            extra={"path": str(path)},
        )

    def connector(self, handle: DestinationHandle, read_only: bool = False) -> BaseConnector:
        return DuckDBConnector({"path": handle.extra["path"], "read_only": read_only})

    async def teardown(self, handle: DestinationHandle) -> None:
        p = Path(handle.extra["path"])
        for suffix in ("", ".wal"):
            f = Path(str(p) + suffix)
            if f.exists():
                f.unlink()

    async def snapshot(self, handle: DestinationHandle, label: str) -> str:
        src = Path(handle.extra["path"])
        dst = self.root / "snapshots" / f"{label}-{uuid.uuid4().hex[:8]}.duckdb"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return str(dst)

    async def restore(self, handle: DestinationHandle, snapshot_id: str) -> None:
        shutil.copyfile(snapshot_id, handle.extra["path"])
