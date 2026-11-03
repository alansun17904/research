#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/cl/u/awsun/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /tmp/zs-nas-awsun-233139
mkdir -p /data/cl/u/awsun/zs-nas/results/post-head
printf '%s\0' graphs/post-head/*.json | xargs -0 -n1 -P4 bash -c '
  set -euo pipefail
  graph=$1
  output=/data/cl/u/awsun/zs-nas/results/post-head/$(basename "$graph")
  test ! -e "$output"
  printf "%s starting %s\n" "$(date -u +%FT%TZ)" "$graph"
  /data/cl/u/awsun/zs-nas/.venv/bin/python evaluate.py --graph "$graph" --max-tokens 8192 --device cuda > "$output.partial" 2> "$output.stderr"
  /data/cl/u/awsun/zs-nas/.venv/bin/python -m json.tool "$output.partial" > /dev/null
  mv "$output.partial" "$output"
  printf "%s complete %s\n" "$(date -u +%FT%TZ)" "$graph"
' _
