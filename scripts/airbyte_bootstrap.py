"""First-run setup of a local abctl Airbyte for ELT-Bench, without the UI.

    python scripts/airbyte_bootstrap.py --elt-bench-root ../ELT-Bench \
        --credential credentials/airbyte_credential.json

Reads the file written from `abctl local credentials` (username, password, client_id,
client_secret), completes the instance setup, records the default workspace id,
publishes ELT-Bench's declarative custom-API source (``setup/elt_snowflake.yaml``) and
records its definition id. Also writes ``setup/airbyte/airbyte_credential.json`` in the
ELT-Bench checkout so ``write_config.py`` can generate task inputs.

Airbyte 2.4.0 only knows declarative CDK majors 0 to 4; the manifest declares 6.33.4.
Run the SQL in docs/OFFICIAL.md (section "Deviations") first, or pass
--register-cdk6 to have this script do it through kubectl inside the abctl container.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import yaml

API = "http://localhost:8000/api/v1"
PUBLIC = "http://localhost:8000/api/public/v1"


def call(url, body=None, token=None, method="POST"):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def register_cdk6(image_version: str = "6.33.4") -> None:
    digest = subprocess.run(["docker", "manifest", "inspect", f"airbyte/source-declarative-manifest:{image_version}"],
                            capture_output=True, text=True, check=True).stdout
    d = json.loads(digest)
    ms = d.get("manifests", [])
    sha = (next((m for m in ms if m.get("platform", {}).get("architecture") == "arm64"), ms[0])["digest"] if ms
           else d["config"]["digest"]).removeprefix("sha256:")
    sql = ("insert into declarative_manifest_image_version (major_version, image_version, image_sha, created_at, updated_at) "
           f"values (6, '{image_version}', '{sha}', now(), now()) on conflict (major_version) do update set image_version=excluded.image_version, image_sha=excluded.image_sha")
    subprocess.run(["docker", "exec", "airbyte-abctl-control-plane", "kubectl", "exec", "-n", "airbyte-abctl", "airbyte-db-0",
                    "--", "psql", "-U", "airbyte", "-d", "db-airbyte", "-c", sql], check=True)
    print("registered declarative CDK major 6 ->", image_version)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elt-bench-root", required=True)
    ap.add_argument("--credential", required=True)
    ap.add_argument("--register-cdk6", action="store_true")
    args = ap.parse_args()
    cred_path = Path(args.credential)
    cred = json.loads(cred_path.read_text())

    if args.register_cdk6:
        register_cdk6()

    st, tok = call(f"{API}/applications/token", {"client_id": cred["client_id"], "client_secret": cred["client_secret"],
                                                 "grant-type": "client_credentials"})
    assert st == 200, tok
    token = tok["access_token"]

    st, cfg = call(f"{API}/instance_configuration", token=token, method="GET")
    if isinstance(cfg, dict) and not cfg.get("initialSetupComplete"):
        st, r = call(f"{API}/instance_configuration/setup", {"email": cred["username"], "anonymousDataCollection": False,
                                                            "initialSetupComplete": True, "displaySetupWizard": False,
                                                            "organizationName": "elt-rlvr"}, token=token)
        print("instance setup:", st)

    st, ws = call(f"{PUBLIC}/workspaces", token=token, method="GET")
    assert st == 200, ws
    wid = ws["data"][0]["workspaceId"]
    cred["workspace_id"] = wid

    st, defs = call(f"{PUBLIC}/workspaces/{wid}/definitions/declarative_sources", token=token, method="GET")
    existing = {d["name"]: d for d in defs.get("data", [])} if isinstance(defs, dict) else {}
    if "elt_bench" in existing:
        cred["api_definition_id"] = existing["elt_bench"]["id"]
        print("custom API source already published")
    else:
        manifest = yaml.safe_load((Path(args.elt_bench_root) / "setup/elt_snowflake.yaml").read_text())
        st, r = call(f"{PUBLIC}/workspaces/{wid}/definitions/declarative_sources", {"name": "elt_bench", "manifest": manifest}, token=token)
        assert st == 200, f"publish failed ({st}): {r}. If 409: register CDK major 6 first (--register-cdk6)."
        cred["api_definition_id"] = r["id"]
        print("published custom API source")

    cred_path.write_text(json.dumps(cred, indent=2))
    cred_path.chmod(0o600)
    dst = Path(args.elt_bench_root) / "setup/airbyte/airbyte_credential.json"
    dst.write_text(json.dumps({k: cred[k] for k in ("username", "password", "api_definition_id", "workspace_id")}, indent=2))
    dst.chmod(0o600)
    print("wrote", cred_path, "and", dst, "(ids only shown):", {"workspace_id": wid, "api_definition_id": cred["api_definition_id"]})


if __name__ == "__main__":
    main()
