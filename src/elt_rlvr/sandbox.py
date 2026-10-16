"""Per-rollout execution sandboxes implementing tinker-cookbook's ``SandboxInterface``.

- ``LocalSandbox``: a subprocess rooted in a throwaway workspace directory with a
  scrubbed environment. Used by the credential-free integration test. It is an
  isolation of *state* (fresh dir, own HOME, own dbt profiles), not a security
  boundary; the grader's ground truth lives outside it and the warehouse file is
  fingerprinted, which is what the reward relies on.
- ``DockerSandbox``: ``docker exec`` into a container started from the ELT-Bench
  ``elt-swe`` image on the benchmark network. Used for official tasks so the agent
  reaches Airbyte, Postgres, MongoDB and LocalStack by their compose hostnames.
  Can be swapped for ``tinker_cookbook.sandbox.modal_sandbox.ModalSandbox``.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MAX_OUTPUT = 16 * 1024
_WORKSPACE_TOKEN = re.compile(r"(?<![\w/])/workspace(?=/|\b)")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    metrics: dict = field(default_factory=dict)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2):]
    return f"{head}\n...[{len(text) - limit} bytes truncated]...\n{tail}"


class LocalSandbox:
    """The agent addresses its workspace as ``/workspace`` (as in the Docker image);
    locally that prefix is rewritten to the real directory in every command."""

    agent_root = "/workspace"

    def __init__(self, workdir: str | Path, env_overrides: dict[str, str] | None = None):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._id = uuid.uuid4().hex[:12]
        home = self.workdir / ".home"
        home.mkdir(exist_ok=True)
        venv_bin = Path(sys.executable).parent
        # Prefer GNU userland when available so `sed -i`, `date`, `readlink` behave as in
        # the Linux agent image (macOS ships BSD variants with different flags).
        gnubin = [d for d in ("/opt/homebrew/opt/gnu-sed/libexec/gnubin", "/opt/homebrew/opt/coreutils/libexec/gnubin",
                              "/usr/local/opt/gnu-sed/libexec/gnubin", "/usr/local/opt/coreutils/libexec/gnubin") if Path(d).is_dir()]
        self.env = {
            "PATH": ":".join([str(venv_bin), *gnubin, "/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin"]),
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
            "DBT_PROFILES_DIR": str(self.workdir / ".dbt"),
            "WORKSPACE": str(self.workdir),
        }
        if env_overrides:
            self.env.update(env_overrides)

    @property
    def sandbox_id(self) -> str:
        return f"local-{self._id}"

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60,
                          max_output_bytes: int | None = None) -> SandboxResult:
        cwd = self._resolve(workdir) if workdir else self.workdir
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT
        command = _WORKSPACE_TOKEN.sub(str(self.workdir), command)
        proc = await asyncio.create_subprocess_shell(
            command, cwd=str(cwd), env=self.env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            code = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            out, err = b"", f"command timed out after {timeout}s".encode()
            code = 124
        return SandboxResult(_cap(out.decode(errors="replace"), limit), _cap(err.decode(errors="replace"), limit), code)

    def _resolve(self, path: str) -> Path:
        # The agent sees its workspace as /workspace; map that prefix onto the real dir.
        if path.startswith("/workspace"):
            return (self.workdir / path[len("/workspace"):].lstrip("/")).resolve()
        return Path(path)

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        p = self._resolve(path)
        if not p.is_file():
            return SandboxResult("", f"no such file: {path}", 1)
        data = p.read_bytes()
        if max_bytes:
            data = data[:max_bytes]
        return SandboxResult(data.decode(errors="replace"), "", 0)

    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        if executable:
            p.chmod(0o755)
        return SandboxResult("", "", 0)

    async def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


class DockerSandbox:
    """Workspace bind-mounted into a long-lived container (ELT-Bench style)."""

    agent_root = "/workspace"

    def __init__(self, workdir: str | Path, image: str = "elt-swe", network: str = "elt-docker_elt_network",
                 container_workdir: str = "/workspace", env: dict[str, str] | None = None):
        self.env = dict(env or {})
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.network = network
        self.container_workdir = container_workdir
        self.container = f"elt-rlvr-{uuid.uuid4().hex[:10]}"
        self._started = False

    @property
    def sandbox_id(self) -> str:
        return self.container

    async def _docker(self, *args: str, timeout: int = 120, stdin: bytes | None = None) -> SandboxResult:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
            return SandboxResult(out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode or 0)
        except asyncio.TimeoutError:
            proc.kill()
            return SandboxResult("", f"docker command timed out after {timeout}s", 124)

    async def start(self) -> None:
        if self._started:
            return
        r = await self._docker(
            "run", "-d", "--rm", "--name", self.container, "--network", self.network,
            "-v", f"{self.workdir}:{self.container_workdir}", "-w", self.container_workdir,
            self.image, "sleep", "infinity",
        )
        if r.exit_code != 0:
            raise RuntimeError(f"docker run failed: {r.stderr}")
        self._started = True

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60,
                          max_output_bytes: int | None = None) -> SandboxResult:
        await self.start()
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT
        env_flags: list[str] = []
        for k, v in self.env.items():
            env_flags += ["-e", f"{k}={v}"]
        r = await self._docker("exec", *env_flags, "-w", workdir or self.container_workdir, self.container,
                               "bash", "-lc", command, timeout=timeout)
        return SandboxResult(_cap(r.stdout, limit), _cap(r.stderr, limit), r.exit_code)

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        r = await self.run_command(f"cat {path}", timeout=timeout, max_output_bytes=max_bytes)
        return r

    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult:
        await self.start()
        data = content if isinstance(content, bytes) else content.encode()
        r = await self._docker("exec", "-i", self.container, "bash", "-c",
                               f"mkdir -p $(dirname {path}) && cat > {path}" + (f" && chmod +x {path}" if executable else ""),
                               stdin=data, timeout=timeout)
        return SandboxResult(r.stdout, r.stderr, r.exit_code)

    async def cleanup(self) -> None:
        if self._started:
            await self._docker("rm", "-f", self.container)
            self._started = False
        shutil.rmtree(self.workdir, ignore_errors=True)
