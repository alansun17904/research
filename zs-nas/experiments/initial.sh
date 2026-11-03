#!/usr/bin/env bash
set -euo pipefail
cd /data/cl/u/awsun/zs-nas
export HF_HOME=/data/cl/u/awsun/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
stage=/tmp/zs-nas-awsun-233139
python=/data/cl/u/awsun/zs-nas/.venv/bin/python
results=/data/cl/u/awsun/zs-nas/results
mkdir -p "$stage" "$results/screen"
rsync -a --exclude .venv --exclude results --exclude __pycache__ ./ "$stage/"
cd "$stage"
"$python" evaluate.py --self-check > "$results/counting-check.txt"
"$python" -m unittest -v > "$results/cluster-tests.log" 2>&1
"$python" -m pip freeze > "$results/requirements-cluster.txt"
"$python" - <<'PY' > "$results/environment.json"
import json, os, platform, torch
from importlib.metadata import version
print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
 'device_count': torch.cuda.device_count(), 'dtype': str(torch.get_default_dtype()),
 'matmul_precision': torch.get_float32_matmul_precision(),
 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
 'packages': {p: version(p) for p in ('transformers','datasets','safetensors')},
 'stage': os.getcwd()}, indent=2))
PY
"$python" - <<'PY'
from pathlib import Path
from model import read_graph
paths=list(Path('graphs/screen').glob('*.json'))
assert len(paths)==108
for path in paths: read_graph(path)
print('Validated all 108 candidate graphs.', flush=True)
PY
for tokens in 100000 8192; do
  output="$results/baseline-$tokens.json"
  test ! -e "$output"
  date -u '+%FT%TZ baseline starting'
  "$python" evaluate.py --graph graphs/baseline.json --max-tokens "$tokens" --device cuda > "$output" 2> "$output.stderr"
  "$python" -m json.tool "$output" >/dev/null
  date -u '+%FT%TZ baseline complete'
done
for graph in graphs/screen/*.json; do
  output="$results/screen/$(basename "$graph")"
  test ! -e "$output"
  printf '%s starting %s\n' "$(date -u '+%FT%TZ')" "$graph"
  "$python" evaluate.py --graph "$graph" --max-tokens 8192 --device cuda > "$output" 2> "$output.stderr"
  "$python" -m json.tool "$output" >/dev/null
  printf '%s complete %s\n' "$(date -u '+%FT%TZ')" "$graph"
done
