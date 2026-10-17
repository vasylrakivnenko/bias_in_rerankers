#!/bin/bash
# Upload code (never .env), install pinned libraries, start both workers.
# Re-running is safe: checkpoints already on the pod or uploaded from the Mac are resumed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
read IP PORT < <(python3 -c "import json;s=json.load(open('$HERE/pod.json'))['ssh'];print(s[0],s[1])")
SSH="ssh -i $HERE/id_ed25519 -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

$SSH root@$IP "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null); mkdir -p /workspace/repo/results/partial /workspace/repo/logs"
rsync -az --exclude __pycache__ -e "$SSH" "$REPO/code/" root@$IP:/workspace/repo/code/
if [ -d "$REPO/results/partial" ]; then   # resume from the Mac's copy of the checkpoints
  rsync -az --include '*.jsonl' --exclude '*' -e "$SSH" "$REPO/results/partial/" root@$IP:/workspace/repo/results/partial/
fi

$SSH root@$IP bash -s <<'REMOTE'
set -e
cd /workspace/repo
pip install -q transformers==5.2.0 sentence-transformers==5.2.3 huggingface_hub==1.4.1 \
  tokenizers==0.22.2 safetensors==0.7.0 einops==0.8.2 2>&1 | tail -2
python -c "import torch,transformers,sentence_transformers as s;print('torch',torch.__version__,'transformers',transformers.__version__,'st',s.__version__,'gpu',torch.cuda.get_device_name(0))"
pkill -f score_open_models.py || true
nohup python code/score_open_models.py --worker big   > logs/big.log   2>&1 &
nohup python code/score_open_models.py --worker small > logs/small.log 2>&1 &
sleep 5; pgrep -fl score_open_models
REMOTE
