#!/bin/bash
# serve_mc_agents.sh -- serve the three Minecraft agents (logger / miner / smith) for eval_mc.sh.
#
# Starts one sglang HTTP server per agent with the engine knobs of the trainer's rollout
# (bf16, triton LLM attention, sdpa vision attention, overlap scheduler off), waits until all three
# answer, warms them up, and writes MODELS_JSON:
#   {"logger": {"url": "http://host:33000", "path": "<dir>"}, "miner": {...}, "smith": {...}}
# Sampling parameters are chosen per call by eval/mc_eval_client.py (the trainer's rollout sampling).
# The script stays in the foreground while the servers run; Ctrl-C stops all three.
#
# Usage:  MC_MODEL_DIR=/path/to/agents MPT_CONDA_ENV=/path/to/env bash scripts/serve_mc_agents.sh
#   MC_MODEL_DIR must hold logger/ miner/ smith/ HF checkpoints (e.g. the merged output of
#   scripts/model_merger.py for a training checkpoint, or Xkev/mc_society_sft for the SFT start).
# Knobs (env, defaults in brackets):
#   MC_MODEL_DIR [required]  BASE_PORT [33000]  MEM_FRAC [0.30]  GPU [0; all three agents share it]
#   MODELS_JSON [$MPT_WORK/mc_eval/models.json]  LOG_DIR [$MPT_WORK/mc_eval/logs]  plus common.sh.
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${MC_MODEL_DIR:?set MC_MODEL_DIR to a directory with logger/ miner/ smith/ checkpoints}"
BASE_PORT=${BASE_PORT:-33000}
MEM_FRAC=${MEM_FRAC:-0.30}
GPU=${GPU:-0}
MODELS_JSON=${MODELS_JSON:-$MPT_WORK/mc_eval/models.json}
LOG_DIR=${LOG_DIR:-$MPT_WORK/mc_eval/logs}
mkdir -p "$(dirname "$MODELS_JSON")" "$LOG_DIR"

DESKS=(logger miner smith)
for d in "${DESKS[@]}"; do
  ls "$MC_MODEL_DIR/$d"/*.safetensors >/dev/null 2>&1 || { echo "missing checkpoint $MC_MODEL_DIR/$d"; exit 1; }
done

HOST=$(hostname -s)
rm -f "$MODELS_JSON"
trap 'rm -f "$MODELS_JSON"; kill 0 2>/dev/null' EXIT
declare -A URL
i=0
for d in "${DESKS[@]}"; do
  port=$((BASE_PORT + i)); i=$((i + 1))
  URL[$d]="http://$HOST:$port"
  echo "[models] $d -> gpu $GPU port $port  $MC_MODEL_DIR/$d"
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    N=0
    while :; do
      N=$((N + 1)); echo "[srv $d] launch #$N $(date +%F_%T)"
      "$PY" -m sglang.launch_server --model-path "$MC_MODEL_DIR/$d" --served-model-name "$d" \
          --host 0.0.0.0 --port "$port" --dtype bfloat16 --mem-fraction-static "$MEM_FRAC" \
          --attention-backend triton --mm-attention-backend sdpa --disable-overlap-schedule \
          --cuda-graph-max-bs 64 --log-level warning
      echo "[srv $d] exit rc=$?; restart in 20 s"
      [ "$N" -ge 20 ] && { echo "[srv $d] giving up"; break; }
      sleep 20
    done
  ) > "$LOG_DIR/srv_${d}.log" 2>&1 &
  sleep 2
done

echo "[models] waiting for three servers (logs in $LOG_DIR)"
ok=0
for t in $(seq 1 120); do
  ok=0
  for d in "${DESKS[@]}"; do
    curl -sf --noproxy '*' --max-time 10 "${URL[$d]}/get_model_info" >/dev/null 2>&1 && ok=$((ok + 1))
  done
  [ "$ok" -eq 3 ] && break
  echo "[models] $ok/3 up ($t)"; sleep 15
done
[ "$ok" -eq 3 ] || { echo "[models] servers never came up -- see $LOG_DIR/srv_*.log"; exit 1; }

# one warm-up generate each, so the first eval call does not pay for CUDA-graph capture
for d in "${DESKS[@]}"; do
  curl -sf --noproxy '*' --max-time 600 "${URL[$d]}/generate" -H 'Content-Type: application/json' \
    -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 4, "temperature": 0}}' >/dev/null \
    && echo "[models] warm $d ok" || echo "[models] warm $d FAILED"
done

"$PY" - "$MODELS_JSON" \
  "logger|${URL[logger]}|$MC_MODEL_DIR/logger" "miner|${URL[miner]}|$MC_MODEL_DIR/miner" \
  "smith|${URL[smith]}|$MC_MODEL_DIR/smith" <<'EOF'
import json, os, sys
out = sys.argv[1]
m = {}
for spec in sys.argv[2:]:
    desk, url, path = spec.split("|", 2)
    m[desk] = {"url": url, "path": path}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(m, f, indent=1)
os.replace(tmp, out)
print(json.dumps(m, indent=1))
EOF
echo "[models] READY $(date +%F_%T) -> $MODELS_JSON"
wait
