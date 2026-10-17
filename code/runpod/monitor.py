"""Sync checkpoints from the pod every 2 minutes and print one line per event.

Events: canary result, each 25% milestone per model (with the running
contrast), failures, stalls (no progress for 15 minutes), worker completion,
and the hard time cap. Each printed line becomes a notification; the script
exits on ALL_DONE or HARD_CAP so the pod can be finalized and terminated.
"""
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
PARTIAL = REPO / "results" / "partial"
SEEN = HERE / "milestones_seen.json"
HARD_CAP_HOURS = 5.0
STALL_MIN = 15
EXPECTED_RUNS = {"big", "small", "big_st6"}   # every status file that must report finished

pod = json.loads((HERE / "pod.json").read_text())
ip, port = pod["ssh"]
ssh = f"ssh -i {HERE / 'id_ed25519'} -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()


def emit(key, line):
    if key not in seen:
        seen.add(key)
        SEEN.write_text(json.dumps(sorted(seen)))
        print(line, flush=True)


while True:
    PARTIAL.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(f"rsync -az -e '{ssh}' root@{ip}:/workspace/repo/results/partial/ {PARTIAL}/",
                       shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SYNC_FAILED {r.stderr.strip()[:200]}", flush=True)
    finished = set()
    for sf in sorted(PARTIAL.glob("status_*.json")):
        st = json.loads(sf.read_text())
        for m, v in st["models"].items():
            short = m.split("/")[-1]
            status = v.get("status", "")
            if v.get("canary"):
                emit(f"{m}|canary", f"CANARY {short}: {v['canary']} -> {status}")
            if status.startswith(("FAILED", "ERROR")):
                emit(f"{m}|fail", f"PROBLEM {short}: {status}")
            total, done = v.get("total") or 0, v.get("done") or 0
            for q in (25, 50, 75, 100):
                if total and done >= total * q / 100:
                    c = v.get("contrast_so_far")
                    emit(f"{m}|{q}", f"{q}% {short}: {done}/{total} queries, contrast "
                         f"{'n/a' if c is None else f'{c:+.1f}'} pts, ties {v.get('tie_pct_so_far')}%, "
                         f"{v.get('rate_q_per_min')} q/min, eta {v.get('eta_min')} min")
            if status == "running":
                age = (time.time() - time.mktime(time.strptime(v["updated"], "%Y-%m-%dT%H:%M:%S"))) / 60
                if age > STALL_MIN:
                    emit(f"{m}|stall|{v['updated']}", f"STALL {short}: no progress for {age:.0f} min")
        if st.get("finished"):
            finished.add(st["worker"])
            emit(f"{sf.name}|finished", f"WORKER_DONE {st['worker']}")
    if finished >= EXPECTED_RUNS:
        print("ALL_DONE", flush=True)
        break
    if (time.time() - pod["created"]) / 3600 > HARD_CAP_HOURS:
        print("HARD_CAP reached -- terminate the pod", flush=True)
        break
    time.sleep(120)
