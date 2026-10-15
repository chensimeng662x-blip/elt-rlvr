"""Snowflake destination for official ELT-Bench tasks.

ELT-Bench stores each task as its own Snowflake database with raw tables and
models under ``AIRBYTE_SCHEMA``. For RL every rollout needs its own copy, so the
logical task name maps to a per-rollout database ``ELT_<task>_<rollout>`` and the
connector rewrites every logical reference accordingly. Evaluation SQL written as
``<task>.<table>`` therefore resolves unchanged, exactly as upstream.

Credentials come from one JSON file (never read by the agent, never logged):

    ELT_RLVR_SNOWFLAKE_CREDENTIAL=/path/to/snowflake_credential.json
    {"account": "...", "user": "...", "password": "...", "role": "ACCOUNTADMIN", "warehouse": "AIRBYTE_WAREHOUSE",
     "agent_user": "AIRBYTE_USER", "agent_password": "...", "agent_role": "AIRBYTE_ROLE"}

The ``agent_*`` values are the ones ELT-Bench's ``setup/destination/setup.sql``
creates; they are what gets written into the agent-visible ``config.yaml``.

Validation status: written against the upstream connector and ``prepare_destination``
logic; exercised end to end only once credentials are available (see VALIDATION.md).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .base import BaseConnector, Destination, DestinationHandle

RAW_SCHEMA = "AIRBYTE_SCHEMA"


def _ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


class SnowflakeConnector(BaseConnector):
    """``database`` arguments are ignored; the logical ``schema`` (task name) is
    mapped through ``database_map`` to the rollout's physical database."""

    def __init__(self, config: dict, database_map: dict[str, str]):
        super().__init__(config)
        self.database_map = {k.lower(): v for k, v in database_map.items()}

    def _db(self, logical: str) -> str:
        return self.database_map.get(logical.lower(), logical)

    def _connect(self):
        import snowflake.connector

        return snowflake.connector.connect(**self.config)

    def _quote(self, ident: str) -> str:
        return ident

    def _norm(self, ident: str) -> str:
        return ident.upper()

    def verify_schema(self, database: str, schema: str) -> bool:
        try:
            df = self.query(f"SELECT schema_name FROM {self._db(schema)}.information_schema.schemata")
        except Exception:  # noqa: BLE001 - database missing
            return False
        return RAW_SCHEMA in {self._norm(x) for x in df.iloc[:, 0].astype(str)}

    def list_tables(self, database: str, schema: str) -> list:
        df = self.query(
            f"SELECT table_name FROM {self._db(schema)}.information_schema.tables "
            f"WHERE table_schema = '{RAW_SCHEMA}' AND table_type = 'BASE TABLE'"
        )
        return [str(x) for x in df.iloc[:, 0]]

    def _fqn(self, database: str, schema: str, table: str) -> str:
        return f"{self._db(schema)}.{RAW_SCHEMA}.{table}"

    def prepare_evaluation_sql(self, sql: str, database: str) -> str:
        return self._rewrite_relations(sql, lambda s, t: f"{self._db(s)}.{RAW_SCHEMA}.{t}")

    def _schemata_sql(self, database: str) -> str:
        return f"SELECT schema_name FROM {self._db(database)}.information_schema.schemata"

    def _tables_sql(self, database: str, schema: str) -> str:
        return (f"SELECT table_name FROM {self._db(schema)}.information_schema.tables "
                f"WHERE table_schema = '{RAW_SCHEMA}' AND table_type = 'BASE TABLE'")


class SnowflakeDestination(Destination):
    name = "snowflake"

    def __init__(self, admin: dict, agent: dict, prefix: str = "ELT"):
        self._admin = admin        # account/user/password/role/warehouse for provisioning + grading
        self._agent = agent        # user/password/role the policy is given
        self.prefix = prefix

    @classmethod
    def from_env(cls) -> "SnowflakeDestination":
        path = os.environ.get("ELT_RLVR_SNOWFLAKE_CREDENTIAL")
        if not path:
            raise RuntimeError("set ELT_RLVR_SNOWFLAKE_CREDENTIAL to the credential JSON path")
        cred = json.loads(Path(path).read_text())
        admin = {k: cred[k] for k in ("account", "user", "password") if k in cred}
        for k in ("role", "warehouse"):
            if cred.get(k):
                admin[k] = cred[k]
        agent = {
            "username": cred.get("agent_user", "AIRBYTE_USER"),
            "password": cred.get("agent_password", cred["password"]),
            "role": cred.get("agent_role", "AIRBYTE_ROLE"),
            "warehouse": cred.get("agent_warehouse", cred.get("warehouse", "AIRBYTE_WAREHOUSE")),
        }
        return cls(admin, agent)

    def _db_name(self, task_name: str, rollout_id: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "_", f"{self.prefix}_{task_name}_{rollout_id}".upper())

    def _exec(self, statements: list[str]) -> None:
        import snowflake.connector

        with snowflake.connector.connect(**self._admin) as conn:
            with conn.cursor() as cur:
                for s in statements:
                    cur.execute(s)

    async def provision(self, task_name: str, rollout_id: str) -> DestinationHandle:
        db = self._db_name(task_name, rollout_id)
        role = _ident(self._agent["role"])
        self._exec([
            f"CREATE OR REPLACE DATABASE {_ident(db)}",
            f"CREATE SCHEMA {_ident(db)}.{RAW_SCHEMA}",
            f"GRANT OWNERSHIP ON SCHEMA {_ident(db)}.{RAW_SCHEMA} TO ROLE {role} COPY CURRENT GRANTS",
            f"GRANT USAGE ON DATABASE {_ident(db)} TO ROLE {role}",
        ])
        agent_cfg = {
            "database": db,
            "schema": RAW_SCHEMA,
            "username": self._agent["username"],
            "password": self._agent["password"],
            "role": self._agent["role"],
            "warehouse": self._agent["warehouse"],
            "account": self._admin["account"],
            "dbt_profile": "elt",
            "dbt_target": "dev",
        }
        profile = {"elt": {"target": "dev", "outputs": {"dev": {
            "type": "snowflake", "account": self._admin["account"], "user": self._agent["username"],
            "password": self._agent["password"], "role": self._agent["role"], "database": db,
            "warehouse": self._agent["warehouse"], "schema": RAW_SCHEMA, "threads": 4,
        }}}}
        return DestinationHandle(self, database=db, schema=task_name, agent_config=agent_cfg,
                                 dbt_profile=profile, extra={"physical_database": db})

    def connector(self, handle: DestinationHandle, read_only: bool = False) -> BaseConnector:
        # Grading and the agent's sql tool both use the admin connection; the sql
        # tool additionally whitelists read-only statements (tools.py).
        return SnowflakeConnector(self._admin, {handle.schema: handle.extra["physical_database"]})

    async def teardown(self, handle: DestinationHandle) -> None:
        self._exec([f"DROP DATABASE IF EXISTS {_ident(handle.extra['physical_database'])}"])

    async def snapshot(self, handle: DestinationHandle, label: str) -> str:
        snap = re.sub(r"[^A-Z0-9_]", "_", f"{handle.extra['physical_database']}_SNAP_{label}".upper())
        self._exec([f"CREATE OR REPLACE DATABASE {_ident(snap)} CLONE {_ident(handle.extra['physical_database'])}"])
        return snap

    async def restore(self, handle: DestinationHandle, snapshot_id: str) -> None:
        db = handle.extra["physical_database"]
        self._exec([f"CREATE OR REPLACE DATABASE {_ident(db)} CLONE {_ident(snapshot_id)}",
                    f"GRANT OWNERSHIP ON SCHEMA {_ident(db)}.{RAW_SCHEMA} TO ROLE {_ident(self._agent['role'])} COPY CURRENT GRANTS",
                    f"GRANT USAGE ON DATABASE {_ident(db)} TO ROLE {_ident(self._agent['role'])}"])
