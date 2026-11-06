#!/bin/bash
# The de-identification utility test on whole résumés (code/utility_whole_resume.py).
#
#   code/run_utility_whole.sh prepare
#   code/run_utility_whole.sh canary|full <model>
#
# "prepare" runs the name detector over every résumé of the design once, so the scoring runs
# share its cached spans and none of them runs it beside a local model. "canary" scores the
# first two queries' first two sets into results/utility_whole/canary_<model>.*; "full" scores
# all 240 sets, resuming from its checkpoint after an interruption, up to three tries.
#
# Settings used (November 2026), the same as the whole-résumé runs (code/run_length_pronoun.sh):
#   voyage-rerank-3, cohere-v5-pro   hosted (keys in .env)
#   bge-m3, gte-modernbert, granite-r2   on the Mac, MPS watermarks below
set -u
cd "$(dirname "$0")/.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
PY=${PY:-venv/bin/python}
mode=$1
mkdir -p logs results/utility_whole results/partial
if [ "$mode" = prepare ]; then
  $PY code/utility_whole_resume.py --prepare > logs/util_prepare.log 2>&1
  echo "== prepare exit $?"
  exit
fi
m=$2; shift 2
mac=(--device mps --max-length 8192 --token-budget 8192 --attention-budget 25000000)
case $m in
  bge-m3)         margs=("${mac[@]}") ;;
  gte-modernbert) margs=("${mac[@]}" --revision f7481e6055501a30fb19d090657df9ec1f79ab2c) ;;
  granite-r2)     margs=("${mac[@]}" --revision d09d3d6971b689bf9c23839e45a470874d46e13a) ;;
  voyage-rerank-3|cohere-v5-pro) margs=() ;;
  *) echo "no settings for $m"; exit 2 ;;
esac
case $mode in
  canary)
    $PY code/utility_whole_resume.py --reranker "$m" --canary ${margs[@]+"${margs[@]}"} "$@" \
        > "logs/util_canary_$m.log" 2>&1
    echo "== canary $m exit $?" ;;
  full)
    [ -f "results/utility_whole/$m.json" ] && { echo "== $m already done"; exit; }
    for attempt in 1 2 3; do
      $PY code/utility_whole_resume.py --reranker "$m" --resume ${margs[@]+"${margs[@]}"} "$@" \
          >> "logs/util_$m.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && break
    done
    echo "== $m done rc=$rc $(date +%H:%M:%S)" ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
