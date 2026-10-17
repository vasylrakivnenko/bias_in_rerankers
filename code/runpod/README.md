# RunPod scoring (open-weight rerankers)

How the ten open-weight audit models and the two self-hosted checks were scored on one
rented NVIDIA RTX 4090 (October 2026). `start.sh` records the exact library pins.

1. `ssh-keygen -t ed25519 -N "" -f code/runpod/id_ed25519` (git-ignored; never commit it).
2. `venv/bin/python code/runpod/pod.py create` -- needs `RUNPOD_API_KEY` in `.env`; the pod
   gets only the public key and the `code/` folder, never `.env`.
3. `bash code/runpod/start.sh` -- installs the pinned libraries and starts the two workers
   of `code/score_open_models.py` (canary-gated, checkpointed per query).
4. `venv/bin/python code/runpod/monitor.py` -- syncs checkpoints every 2 minutes and prints
   canaries, 25% milestones, failures and stalls.
5. `venv/bin/python code/runpod/pod.py terminate`, then
   `venv/bin/python code/score_open_models.py --finalize`.

zerank-2 and mxbai-rerank-large-v2 need sentence-transformers 6.x (they ship LogitScore
modules); they were run in a separate environment with sentence-transformers 6.1.0.
