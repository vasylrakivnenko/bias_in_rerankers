#!/bin/bash
# The companion experiment's length x pronoun runs (results/length_pronoun/).
#
#   code/run_length_pronoun.sh canary|full <model> [extra realistic_pool_experiment.py args]
#
# Twelve conditions per reranker, on the banked pools (--expect-pool-sha256) with only the
# résumé owner's own name and pronouns changed (--marker-labels): the text after the
# prepended name cut to 35, 75, 150, 300 or 600 words, or whole (--cut-words 0), each with
# the name alone ("name") or followed by OWNER_LINE ("line", --owner-line). A finished
# condition is skipped; an interrupted one resumes from its checkpoint, up to three tries.
# "canary" scores two pools of the two extreme conditions into canary_* files.
#
# Settings used (October 2026):
#   voyage-rerank-3, voyage-rerank-2, cohere-v5-pro, qwen3-8b   (hosted; keys in .env)
#   bge-m3           on the Mac: --device mps --max-length 8192 --token-budget 8192
#                    --attention-budget 25000000 (with the MPS watermarks below)
#   mxbai-large-v2   on an RTX 4090 (code/runpod/): --device cuda --max-length 8192
#                    --token-budget 8192 --batch-size 1 --model-dtype bfloat16
#                    --revision ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6
# Then: venv/bin/python code/companion_whole_resume.py
set -u
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
PY=${PY:-venv/bin/python}
SHA=91687571c4c9dd4f78c5d442c751070ea0efc18b56e5f9713479a4532f80c400
mode=$1; m=$2; shift 2
case $m in
  bge-m3) margs=(--device mps --max-length 8192 --token-budget 8192 --attention-budget 25000000) ;;
  mxbai-large-v2) margs=(--max-length 8192 --token-budget 8192 --batch-size 1
                         --model-dtype bfloat16 --revision ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6) ;;
  *) margs=(--max-length 32000) ;;
esac
common=(--rerankers "$m" --pools-per-query 8 --span-cache results/.span_cache_6000.json
        --expect-pool-sha256 $SHA --marker-labels data/owner_marker_labels.json "${margs[@]}" "$@")
mkdir -p results/length_pronoun results/partial logs
if [ "$mode" = canary ]; then conds="35:line 0:name"; else
  conds="35:name 35:line 0:name 0:line 150:name 150:line 600:name 600:line 75:name 75:line 300:name 300:line"; fi
for c in $conds; do
  n=${c%%:*}; arm=${c##*:}
  tag=w$([ "$n" = 0 ] && echo full || echo "$n")_$arm
  extra=(--cut-words "$n"); [ "$arm" = line ] && extra+=(--owner-line)
  if [ "$mode" = canary ]; then
    $PY code/realistic_pool_experiment.py "${common[@]}" "${extra[@]}" --limit-pools 2 \
        --out "results/length_pronoun/canary_${m}_$tag.json" \
        --checkpoint "results/partial/lp_canary_${m}_$tag.checkpoint.json" \
        > "logs/lp_canary_${m}_$tag.log" 2>&1
    echo "== canary $m $tag exit $?"
    continue
  fi
  out=results/length_pronoun/${m}_$tag.json
  [ -f "$out" ] && { echo "== $m $tag already done"; continue; }
  for attempt in 1 2 3; do
    $PY code/realistic_pool_experiment.py "${common[@]}" "${extra[@]}" --resume --out "$out" \
        >> "logs/lp_$m.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
  done
  echo "== $m $tag done rc=$rc $(date +%H:%M:%S)"
done
