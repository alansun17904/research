#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/data/cl/u/awsun/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd /tmp/zs-nas-awsun-233139
mkdir -p /data/cl/u/awsun/zs-nas/results/full
printf '%s\0' graphs/screen/repeat-head-l4-h3.json graphs/screen/repeat-head-l4-h5.json graphs/screen/repeat-head-l4-h1.json graphs/screen/repeat-head-l4-h4.json graphs/screen/repeat-head-l3-h5.json graphs/screen/repeat-head-l3-h1.json graphs/screen/repeat-head-l4-h2.json graphs/screen/repeat-head-l3-h6.json graphs/screen/repeat-head-l4-h7.json graphs/screen/repeat-head-l4-h0.json graphs/screen/repeat-head-l4-h6.json graphs/screen/repeat-head-l3-h0.json graphs/screen/repeat-head-l2-h3.json graphs/screen/repeat-head-l5-h0.json graphs/screen/repeat-head-l3-h7.json graphs/screen/repeat-head-l1-h6.json graphs/screen/repeat-head-l1-h0.json graphs/screen/repeat-head-l2-h0.json graphs/screen/repeat-head-l1-h4.json graphs/screen/repeat-head-l5-h1.json graphs/screen/repeat-head-l2-h7.json graphs/screen/repeat-head-l2-h4.json graphs/screen/repeat-head-l2-h1.json graphs/screen/repeat-head-l2-h2.json graphs/screen/repeat-head-l1-h5.json graphs/screen/repeat-head-l2-h6.json graphs/screen/repeat-head-l1-h7.json graphs/screen/repeat-head-l2-h5.json graphs/screen/repeat-head-l3-h2.json graphs/screen/repeat-head-l1-h3.json graphs/screen/repeat-head-l1-h2.json graphs/screen/repeat-head-l3-h4.json graphs/screen/repeat-head-l1-h1.json graphs/screen/repeat-head-l3-h3.json graphs/screen/repeat-head-l0-h2.json graphs/screen/repeat-head-l0-h1.json graphs/screen/repeat-head-l0-h4.json graphs/screen/repeat-head-l0-h0.json graphs/screen/repeat-head-l0-h5.json graphs/screen/repeat-head-l0-h6.json graphs/screen/repeat-head-l0-h7.json graphs/screen/repeat-head-l0-h3.json graphs/screen/repeat-head-l5-h2.json graphs/screen/repeat-head-l5-h3.json graphs/screen/repeat-head-l5-h4.json graphs/screen/repeat-head-l5-h5.json graphs/screen/repeat-head-l5-h6.json graphs/screen/repeat-head-l5-h7.json | xargs -0 -n1 -P4 bash -c '
  set -euo pipefail
  graph=$1
  output=/data/cl/u/awsun/zs-nas/results/full/$(basename "$graph")
  test ! -e "$output"
  printf "%s starting %s\n" "$(date -u +%FT%TZ)" "$graph"
  /data/cl/u/awsun/zs-nas/.venv/bin/python evaluate.py --graph "$graph" --max-tokens 100000 --device cuda > "$output.partial" 2> "$output.stderr"
  /data/cl/u/awsun/zs-nas/.venv/bin/python -m json.tool "$output.partial" > /dev/null
  mv "$output.partial" "$output"
  printf "%s complete %s\n" "$(date -u +%FT%TZ)" "$graph"
' _
