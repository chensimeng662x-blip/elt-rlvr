"""Official EL runtime: Terraform (Airbyte provider 0.6.5) + Airbyte sync jobs.

``run()`` executes, inside the rollout sandbox, ``terraform init`` and
``terraform apply -parallelism=1`` on ``/workspace/elt`` (the directory ELT-Bench
tasks ship ``main.tf`` in), reads the connection ids and stream names from the
resulting ``terraform.tfstate`` on the host side, triggers one sync job per
connection through the Airbyte public API, polls until every job is terminal,
and fingerprints every synced table in the warehouse for provenance.

Host-side Airbyte access:
    ELT_RLVR_AIRBYTE_URL=http://localhost:8000/api/public/v1   (abctl default)
    ELT_RLVR_AIRBYTE_CREDENTIAL=/path/to/airbyte_credential.json  {"username": ..., "password": ...}

Validation status: written against ELT-Bench's ``check_job_status.py`` and
``trigger_job.md``; exercised end to end only once Airbyte is deployed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .base import ELReport, ELRuntime, StreamProvenance, fingerprint_table

TERMINAL = {"succeeded", "failed", "cancelled", "incomplete"}


class AirbyteTerraformRuntime(ELRuntime):
    name = "airbyte_terraform"

    def __init__(self, handle, workdir, sandbox, api_url: str | None = None, credential_path: str | None = None,
                 apply_timeout: int = 900, sync_timeout: int = 1800, poll_seconds: int = 10, sync_delay_seconds: int = 0):
        super().__init__(handle, workdir)
        self.sandbox = sandbox
        self.api_url = (api_url or os.environ.get("ELT_RLVR_AIRBYTE_URL", "http://localhost:8000/api/public/v1")).rstrip("/")
        cred_path = credential_path or os.environ.get("ELT_RLVR_AIRBYTE_CREDENTIAL")
        if not cred_path:
            raise RuntimeError("set ELT_RLVR_AIRBYTE_CREDENTIAL to the Airbyte credential JSON path")
        cred = json.loads(Path(cred_path).read_text())
        token = base64.b64encode(f"{cred['username']}:{cred['password']}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json", "Accept": "application/json"}
        self.apply_timeout = apply_timeout
        self.sync_timeout = sync_timeout
        self.poll_seconds = poll_seconds
        self.sync_delay_seconds = sync_delay_seconds
        self.job_history: list[dict] = []

    # -- Airbyte API -----------------------------------------------------------
    def _api(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(f"{self.api_url}{path}", method=method, headers=self._headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}")

    def _trigger(self, connection_id: str) -> dict:
        return self._api("POST", "/jobs", {"jobType": "sync", "connectionId": connection_id})

    def _job(self, job_id: int | str) -> dict:
        return self._api("GET", f"/jobs/{job_id}")

    # -- Terraform state -------------------------------------------------------
    def _connections_from_state(self) -> list[dict]:
        state_path = Path(self.workdir) / "elt" / "terraform.tfstate"
        if not state_path.exists():
            return []
        state = json.loads(state_path.read_text())
        out = []
        for res in state.get("resources", []):
            if res.get("type") != "airbyte_connection":
                continue
            for inst in res.get("instances", []):
                attrs = inst.get("attributes", {})
                streams = [s.get("name") for s in (attrs.get("configurations") or {}).get("streams", []) if s.get("name")]
                out.append({"connection_id": attrs.get("connection_id"), "name": attrs.get("name"), "streams": streams})
        return [c for c in out if c["connection_id"]]

    # -- main --------------------------------------------------------------------
    async def run(self) -> ELReport:
        init = await self.sandbox.run_command("cd /workspace/elt && terraform init -input=false -no-color", timeout=self.apply_timeout)
        if init.exit_code != 0:
            return ELReport(False, {"status": "terraform_init_failed", "stdout": init.stdout[-4000:], "stderr": init.stderr[-4000:]})
        apply = await self.sandbox.run_command(
            "cd /workspace/elt && terraform apply -auto-approve -input=false -no-color -parallelism=1", timeout=self.apply_timeout)
        if apply.exit_code != 0:
            return ELReport(False, {"status": "terraform_apply_failed", "stdout": apply.stdout[-4000:], "stderr": apply.stderr[-4000:]})

        connections = self._connections_from_state()
        if not connections:
            return ELReport(False, {"status": "no_connections", "hint": "terraform apply succeeded but created no airbyte_connection resources"})

        jobs: dict[str, dict] = {}
        for i, c in enumerate(connections):
            if i and self.sync_delay_seconds:
                await asyncio.sleep(self.sync_delay_seconds)
            try:
                j = await asyncio.to_thread(self._trigger, c["connection_id"])
                jobs[c["connection_id"]] = {"job_id": j.get("jobId"), "status": j.get("status", "pending"), "connection": c}
            except urllib.error.HTTPError as e:
                jobs[c["connection_id"]] = {"job_id": None, "status": "failed", "error": f"trigger failed: {e.code} {e.read().decode()[:500]}", "connection": c}

        deadline = time.time() + self.sync_timeout
        while time.time() < deadline:
            pending = [cid for cid, j in jobs.items() if j["job_id"] is not None and j["status"] not in TERMINAL]
            if not pending:
                break
            for cid in pending:
                try:
                    j = await asyncio.to_thread(self._job, jobs[cid]["job_id"])
                    jobs[cid]["status"] = j.get("status", jobs[cid]["status"])
                    jobs[cid]["records_synced"] = j.get("rowsSynced") or j.get("recordsSynced")
                except Exception as e:  # noqa: BLE001
                    jobs[cid]["poll_error"] = str(e)[:300]
            await asyncio.sleep(self.poll_seconds)

        ok = True
        public_jobs = []
        with self.handle.connector(read_only=True) as conn:
            for cid, j in jobs.items():
                entry = {"connection": j["connection"]["name"], "connection_id": cid, "job_id": j["job_id"],
                         "status": j["status"], "streams": j["connection"]["streams"]}
                if "records_synced" in j:
                    entry["records_synced"] = j["records_synced"]
                if "error" in j:
                    entry["error"] = j["error"]
                if j["status"] != "succeeded":
                    ok = False
                else:
                    for stream in j["connection"]["streams"]:
                        try:
                            rows, fp = fingerprint_table(conn, self.handle.database, self.handle.schema, stream)
                            self.provenance[stream.lower()] = StreamProvenance(stream, rows, fp, str(j["job_id"]))
                        except Exception as e:  # noqa: BLE001 - table absent after a "succeeded" job
                            entry.setdefault("warnings", []).append(f"{stream}: {str(e)[:200]}")
                public_jobs.append(entry)
        self.job_history.append({"jobs": public_jobs})
        return ELReport(ok, {"status": "succeeded" if ok else "partial_failure", "jobs": public_jobs}, dict(self.provenance))
