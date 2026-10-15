"""The policy's action space: four tools shared by every destination.

- ``bash``   : run a shell command in the rollout sandbox (workspace at /workspace).
- ``sql``    : read-only query against the rollout's own warehouse namespace.
- ``run_el`` : host-mediated extraction & loading from the configuration the
               policy wrote; the only path that produces provenance.
- ``submit`` : terminal action; grading happens after it.
"""

from __future__ import annotations

import io
import json
import re
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

from .destinations.base import DestinationHandle
from .el_runtime.base import ELRuntime

_READ_ONLY = re.compile(r"^\s*(select|with|describe|show|explain|pragma\s+table_info)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|create|alter|drop|truncate|copy|attach|install|load|call|export|import)\b", re.IGNORECASE)


class ELTTools:
    def __init__(self, sandbox, handle: DestinationHandle, el_runtime: ELRuntime,
                 command_timeout: int = 600, sql_row_limit: int = 50, max_output_chars: int = 16 * 1024):
        self._sandbox = sandbox
        self._handle = handle
        self._el = el_runtime
        self._command_timeout = command_timeout
        self._sql_row_limit = sql_row_limit
        self._max_output = max_output_chars
        self.submitted = False
        self.el_runs = 0
        self.bash_calls = 0
        self.sql_calls = 0

    @tool
    async def bash(self, command: Annotated[str, "Shell command to run in /workspace."]) -> ToolResult:
        """Run a bash command inside the task workspace (/workspace). Use it to read
        the task files, write Terraform/EL configuration and dbt models, and run dbt."""
        self.bash_calls += 1
        r = await self._sandbox.run_command(command, timeout=self._command_timeout,
                                            max_output_bytes=self._max_output)
        return simple_tool_result(json.dumps({"exit_code": r.exit_code, "stdout": r.stdout, "stderr": r.stderr}))

    @tool
    async def sql(self, query: Annotated[str, "A single read-only SQL statement (SELECT / WITH / DESCRIBE / SHOW)."]) -> ToolResult:
        """Run a read-only SQL query against the destination warehouse namespace of
        this task and return up to 50 rows as CSV. Use it to inspect loaded tables
        and check your data models. Writes are rejected; use dbt to create models."""
        self.sql_calls += 1
        q = query.strip().rstrip(";")
        if ";" in q or not _READ_ONLY.match(q) or _FORBIDDEN.search(q):
            return simple_tool_result(json.dumps({"error": "only a single read-only SELECT/WITH/DESCRIBE/SHOW statement is allowed"}))
        try:
            with self._handle.connector(read_only=True) as c:
                df = c.query(q)
        except Exception as e:  # noqa: BLE001
            return simple_tool_result(json.dumps({"error": str(e)[: self._max_output]}))
        total = len(df)
        buf = io.StringIO()
        df.head(self._sql_row_limit).to_csv(buf, index=False)
        text = buf.getvalue()
        if len(text) > self._max_output:
            text = text[: self._max_output] + "\n...[truncated]"
        note = f"\n({total} rows total, showing first {self._sql_row_limit})" if total > self._sql_row_limit else f"\n({total} rows)"
        return simple_tool_result(text + note)

    @tool
    async def run_el(self) -> ToolResult:
        """Execute the extraction-and-loading step from the configuration you wrote in
        /workspace/elt (see the documentation folder for the required format). Returns
        one job per stream with its status and records synced. Re-run after fixing errors."""
        self.el_runs += 1
        try:
            report = await self._el.run()
        except Exception as e:  # noqa: BLE001
            return simple_tool_result(json.dumps({"status": "runtime_error", "error": str(e)[: self._max_output]}))
        return simple_tool_result(json.dumps(report.public)[: self._max_output])

    @tool
    async def submit(self) -> ToolResult:
        """Submit the pipeline. Call this once the raw tables are loaded and every data
        model in data_model.yaml exists as a table in the destination. Ends the episode;
        the warehouse state is then graded."""
        self.submitted = True
        return simple_tool_result(json.dumps({"status": "submitted"}), should_stop=True)

    def all(self):
        return [self.bash, self.sql, self.run_el, self.submit]
