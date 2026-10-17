"""Create / inspect / terminate the scoring pod. Usage: pod.py create|info|terminate"""
import json
import sys
import time
from pathlib import Path

import runpod

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATE = HERE / "pod.json"
env = dict(l.split("=", 1) for l in (REPO / ".env").read_text().splitlines() if "=" in l)
runpod.api_key = env["RUNPOD_API_KEY"].strip()


def info(pid):
    p = runpod.get_pod(pid)
    ports = ((p.get("runtime") or {}).get("ports")) or []
    ssh = [x for x in ports if x.get("privatePort") == 22 and x.get("isIpPublic")]
    return p, (ssh[0]["ip"], ssh[0]["publicPort"]) if ssh else None


cmd = sys.argv[1]
if cmd == "create":
    assert not STATE.exists(), f"a pod is already recorded in {STATE}"
    pod = None
    for cloud in ("SECURE", "COMMUNITY"):
        try:
            pod = runpod.create_pod(
                name="biasinrerankers-scoring",
                image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                gpu_type_id="NVIDIA GeForce RTX 4090", cloud_type=cloud, gpu_count=1,
                container_disk_in_gb=120, volume_in_gb=0, ports="22/tcp",
                support_public_ip=True,
                env={"PUBLIC_KEY": (HERE / "id_ed25519.pub").read_text().strip()})
            print("created on", cloud)
            break
        except Exception as e:  # noqa: BLE001
            print(cloud, "unavailable:", str(e)[:200])
    assert pod, "no 4090 available"
    STATE.write_text(json.dumps({"id": pod["id"], "created": time.time()}))
    for _ in range(60):
        p, ssh = info(pod["id"])
        if ssh:
            print(json.dumps({"id": pod["id"], "ssh": ssh, "cost_per_hr": p.get("costPerHr")}))
            STATE.write_text(json.dumps({"id": pod["id"], "created": time.time(), "ssh": ssh,
                                         "cost_per_hr": p.get("costPerHr")}))
            break
        time.sleep(10)
    else:
        print("pod created but SSH not ready yet; run `pod.py info`")
elif cmd == "info":
    st = json.loads(STATE.read_text())
    p, ssh = info(st["id"])
    print(json.dumps({"id": st["id"], "status": p.get("desiredStatus"), "ssh": ssh,
                      "cost_per_hr": p.get("costPerHr"),
                      "hours": round((time.time() - st["created"]) / 3600, 2)}))
elif cmd == "terminate":
    st = json.loads(STATE.read_text())
    runpod.terminate_pod(st["id"])
    time.sleep(5)
    left = [x["id"] for x in runpod.get_pods() if x["id"] == st["id"]]
    print("terminated" if not left else f"STILL PRESENT: {left}")
    if not left:
        STATE.rename(HERE / f"pod_terminated_{st['id']}.json")
