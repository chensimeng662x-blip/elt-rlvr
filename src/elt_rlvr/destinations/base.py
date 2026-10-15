"""Warehouse abstraction.

Grading talks to the warehouse only through ``BaseConnector`` (the same
interface ELT-Bench's evaluators use: ``verify_schema``, ``list_tables``,
``table_size``, ``fetch_table``, ``query_evaluation_sql``). A ``Destination``
adds the per-rollout lifecycle around it. Adding a warehouse means
implementing these two classes; nothing in the rollout, tools, grading or
reward code changes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_RELATION_PATTERN = re.compile(
    r"\b(from|join)\s+([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)", re.IGNORECASE
)


class BaseConnector:
    """Port of ELT-Bench ``evaluation/db_connectors.BaseConnector``."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self._conn = None

    def _connect(self):
        raise NotImplementedError

    def _quote(self, ident: str) -> str:
        raise NotImplementedError

    def _norm(self, ident: str) -> str:
        raise NotImplementedError

    def _schemata_sql(self, database: str) -> str:
        raise NotImplementedError

    def _tables_sql(self, database: str, schema: str) -> str:
        raise NotImplementedError

    def conn(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def query(self, sql: str) -> pd.DataFrame:
        cur = self.conn().cursor()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        finally:
            cur.close()
        return pd.DataFrame(rows, columns=cols)

    def prepare_evaluation_sql(self, sql: str, database: str) -> str:
        return sql

    def query_evaluation_sql(self, sql: str, database: str) -> pd.DataFrame:
        return self.query(self.prepare_evaluation_sql(sql, database))

    @staticmethod
    def _rewrite_relations(sql: str, qualify) -> str:
        def replace(match):
            keyword, schema, table = match.groups()
            return f"{keyword} {qualify(schema, table)}"

        return _RELATION_PATTERN.sub(replace, sql)

    def _fqn(self, database: str, schema: str, table: str) -> str:
        return f"{self._quote(database)}.{self._quote(schema)}.{self._quote(table)}"

    def verify_schema(self, database: str, schema: str) -> bool:
        df = self.query(self._schemata_sql(database))
        names = {self._norm(x) for x in df.iloc[:, 0].astype(str)}
        return self._norm(schema) in names

    def list_tables(self, database: str, schema: str) -> list:
        df = self.query(self._tables_sql(database, schema))
        return [str(x) for x in df.iloc[:, 0]]

    def list_columns(self, database: str, schema: str, table: str) -> list[str]:
        df = self.query(f"SELECT * FROM {self._fqn(database, schema, table)} WHERE 1=0")
        return [str(c) for c in df.columns]

    def table_size(self, database: str, schema: str, table: str) -> int:
        df = self.query(f"SELECT COUNT(*) FROM {self._fqn(database, schema, table)}")
        return int(df.iloc[0, 0])

    def fetch_table(self, database: str, schema: str, table: str) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {self._fqn(database, schema, table)}")


@dataclass
class DestinationHandle:
    """One rollout's namespace in a warehouse.

    ``database``/``schema`` are the logical names the ELT-Bench evaluators use
    (``table_size(database, schema, table)``). ``agent_config`` is what gets
    written into the agent-visible ``config.yaml`` destination block.
    """

    destination: "Destination"
    database: str
    schema: str
    agent_config: dict[str, Any]
    dbt_profile: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def connector(self, read_only: bool = False) -> BaseConnector:
        return self.destination.connector(self, read_only=read_only)


class Destination(ABC):
    """Lifecycle of per-rollout namespaces in one warehouse type."""

    name: str = "abstract"

    @abstractmethod
    async def provision(self, task_name: str, rollout_id: str) -> DestinationHandle:
        """Create an isolated, empty namespace for one rollout."""

    @abstractmethod
    def connector(self, handle: DestinationHandle, read_only: bool = False) -> BaseConnector:
        """Fresh connector; caller closes it. ``read_only`` is used by the agent's sql tool."""

    @abstractmethod
    async def teardown(self, handle: DestinationHandle) -> None:
        """Drop the namespace and any credentials created for it."""

    async def snapshot(self, handle: DestinationHandle, label: str) -> str:
        """Optional: persist the current namespace state (used for transform-only starts)."""
        raise NotImplementedError(f"{self.name} does not support snapshots")

    async def restore(self, handle: DestinationHandle, snapshot_id: str) -> None:
        raise NotImplementedError(f"{self.name} does not support snapshots")
