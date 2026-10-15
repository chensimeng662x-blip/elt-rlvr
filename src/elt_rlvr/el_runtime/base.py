"""Extract-and-load runtimes.

The agent configures EL by writing files in its workspace; it executes EL by
calling the host-mediated ``run_el`` tool. The runtime that performs the load
also records *provenance* (what it loaded, with a content fingerprint taken
from the warehouse right after loading). The grader later re-fingerprints the
raw tables: a table that matches the expected row count but not the
fingerprint was written by something other than the sanctioned EL path and
earns no credit.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..destinations.base import BaseConnector, DestinationHandle

AIRBYTE_META_COLUMNS = ("_airbyte_raw_id", "_airbyte_extracted_at", "_airbyte_meta", "_airbyte_generation_id")


@dataclass
class StreamProvenance:
    table: str
    rows: int
    fingerprint: str
    job_id: str
    status: str = "succeeded"


@dataclass
class ELReport:
    """What ``run_el`` returns to the policy (``public``) and what only the grader keeps."""

    ok: bool
    public: dict[str, Any]
    provenance: dict[str, StreamProvenance] = field(default_factory=dict)


def fingerprint_table(connector: BaseConnector, database: str, schema: str, table: str) -> tuple[int, str]:
    """Order-independent content hash of a warehouse table, ignoring Airbyte metadata columns."""
    df = connector.fetch_table(database, schema, table)
    keep = [c for c in df.columns if c.lower() not in AIRBYTE_META_COLUMNS]
    df = df[sorted(keep, key=str.lower)]
    # pandas keeps missing values as NaN through astype(str); normalise explicitly.
    cells = df.map(lambda v: "\x00" if pd.isna(v) else str(v))
    rows = sorted(",".join(r) for r in cells.itertuples(index=False, name=None))
    h = hashlib.sha256()
    h.update(",".join(c.lower() for c in df.columns).encode())
    for r in rows:
        h.update(b"\n")
        h.update(r.encode())
    return len(df), h.hexdigest()


class ELRuntime(ABC):
    """One instance per rollout; holds provenance for that rollout only."""

    name: str = "abstract"

    def __init__(self, handle: DestinationHandle, workdir):
        self.handle = handle
        self.workdir = workdir
        self.provenance: dict[str, StreamProvenance] = {}

    @abstractmethod
    async def run(self) -> ELReport:
        """Read the agent's EL configuration from the workspace and execute it."""

    def agent_documentation(self) -> dict[str, str]:
        """Extra documentation files (name -> markdown) to expose to the agent."""
        return {}
