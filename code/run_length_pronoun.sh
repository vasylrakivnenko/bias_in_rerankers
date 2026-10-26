#!/bin/bash
# The companion experiment's owner-only runs (results/length_pronoun/).
#
#   code/run_length_pronoun.sh canary|full|whole-canary|whole <model> [extra realistic_pool_experiment.py args]
#
# Every run scores the banked pools (--expect-pool-sha256) with only the résumé owner's own
# name and pronouns changed (--marker-labels). "full": the length x pronoun design, twelve
# conditions -- the text after the prepended name cut to 35, 75, 150, 300 or 600 words, or
# whole (--cut-words 0), each with the name alone ("name") or followed by OWNER_LINE ("line",
# --owner-line). "whole": the whole-résumé name-only condition alone, for every reranker of
# the whole-résumé table; jina-reranker-v3 is listwise and order-sensitive, so it also gets a
# female-first pass (--female-first). A finished condition is skipped; an interrupted one
# resumes from its checkpoint, up to three tries. The canary modes score two pools of
# their first and last conditions into canary_* files.
#
# Settings used (October 2026):
#   hosted (keys in .env): --max-length 32000
#   bge-m3, gte-modernbert, granite-r2, jina   on the Mac (MPS watermarks below)
#   zerank-2, mxbai-large-v2, qwen3-0.6b, qwen3-4b, nemotron-1b, jina-v3
#                    on an RTX 4090 (code/runpod/), bfloat16, one input per batch, the
#                    audit's pinned revision, sentence-transformers 6.1.0; with
#                    HF_HUB_OFFLINE=1, cache jina-v3's default branch too (its code loads
#                    the tokenizer from there; it is the pinned commit, d7d7e73b)
# Then: venv/bin/python code/companion_whole_resume.py
set -u
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
PY=${PY:-venv/bin/python}
SHA=91687571c4c9dd4f78c5d442c751070ea0efc18b56e5f9713479a4532f80c400
mode=$1; m=$2; shift 2
mac=(--device mps --max-length 8192 --token-budget 8192 --attention-budget 25000000)
gpu=(--device cuda --max-length 8192 --token-budget 8192 --batch-size 1 --model-dtype bfloat16)
case $m in
  bge-m3)         margs=("${mac[@]}") ;;
  gte-modernbert) margs=("${mac[@]}" --revision f7481e6055501a30fb19d090657df9ec1f79ab2c) ;;
  granite-r2)     margs=("${mac[@]}" --revision d09d3d6971b689bf9c23839e45a470874d46e13a) ;;
  jina)           margs=(--device mps --max-length 1024 --token-budget 8192) ;;
  mxbai-large-v2) margs=("${gpu[@]}" --revision ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6) ;;
  zerank-2)       margs=("${gpu[@]}" --revision 5eae30d5ee3c6b2df2ef6d723bde45172d761c4c) ;;
  qwen3-0.6b)     margs=("${gpu[@]}" --revision e61197ed45024b0ed8a2d74b80b4d909f1255473) ;;
  qwen3-4b)       margs=("${gpu[@]}" --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5) ;;
  nemotron-1b)    margs=("${gpu[@]}" --revision 828765652b05bd439c9789d2a6d093db1caa1443) ;;
  jina-v3)        margs=(--device cuda --max-length 8192) ;;   # listwise; revision pinned in code
  *)              margs=(--max-length 32000) ;;
esac
common=(--rerankers "$m" --pools-per-query 8 --span-cache results/.span_cache_6000.json
        --expect-pool-sha256 $SHA --marker-labels data/owner_marker_labels.json "${margs[@]}" "$@")
mkdir -p results/length_pronoun results/partial logs
case $mode in
  full)         conds="35:name 35:line 0:name 0:line 150:name 150:line 600:name 600:line 75:name 75:line 300:name 300:line" ;;
  canary)       conds="35:line 0:name" ;;
  whole|whole-canary)
                conds="0:name"; [ "$m" = jina-v3 ] && conds="0:name 0:name:ff" ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
for c in $conds; do
  IFS=: read -r n arm order <<< "$c"
  tag=w$([ "$n" = 0 ] && echo full || echo "$n")_$arm
  extra=(--cut-words "$n"); [ "$arm" = line ] && extra+=(--owner-line)
  [ "$order" = ff ] && { tag=${tag}_female_first; extra+=(--female-first); }
  if [ "${mode%canary}" != "$mode" ]; then
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
