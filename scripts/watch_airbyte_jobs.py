"""Poll Airbyte and print one line whenever a job's status/attempt count changes.
Exits when every job is terminal. Used for operator monitoring only."""
import json, sys, time, urllib.request, pathlib
cred = json.loads((pathlib.Path(__file__).resolve().parents[1] / "credentials/airbyte_credential.json").read_text())
def call(url, body=None, token=None, method="POST"):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type":"application/json","Accept":"application/json", **({"Authorization":f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read().decode() or "{}")
def token():
    return call("http://localhost:8000/api/v1/applications/token", {"client_id": cred["client_id"], "client_secret": cred["client_secret"], "grant-type": "client_credentials"})["access_token"]
prev = {}
tok = token(); t0 = time.time()
while True:
    try:
        if time.time() - t0 > 540: tok = token(); t0 = time.time()
        conns = {c["connectionId"]: c["name"] for c in call("http://localhost:8000/api/public/v1/connections?limit=100", token=tok, method="GET").get("data", [])}
        jobs = call("http://localhost:8000/api/public/v1/jobs?limit=20", token=tok, method="GET").get("data", [])
        all_terminal = True
        for j in jobs:
            name = conns.get(j["connectionId"], "?")
            st = j["status"]
            info = call("http://localhost:8000/api/v1/jobs/get", {"id": j["jobId"]}, tok)
            att = ",".join(a["attempt"]["status"] for a in info["attempts"])
            key = f"{name}|{st}|{att}"
            if prev.get(name) != key:
                print(f"{time.strftime('%H:%M:%S')} {name}: {st} rows={j.get('rowsSynced')} attempts=[{att}]", flush=True)
                prev[name] = key
            if st not in ("succeeded", "failed", "cancelled"): all_terminal = False
        if all_terminal and jobs:
            print("ALL JOBS TERMINAL", flush=True); break
    except Exception as e:
        print(f"poll error: {str(e)[:120]}", flush=True)
    time.sleep(30)
