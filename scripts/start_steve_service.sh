#!/bin/bash
# start_steve_service.sh -- optional shared STEVE-1 inference service (src/server/mc_env/steve_service.py).
#
# By default every env server loads STEVE-1 on its own GPU. To host several env servers on CPU-only
# machines, run this service on one GPU and start the env servers with MC_STEVE_URL=http://<host>:<port>
# (and MC_DEVICE=cpu); every slot then sends its frames here and the service batches them per forward.
#
# Usage:  MC_SIM_ENV=/path/to/conda/envs/minestudio bash scripts/start_steve_service.sh
# Knobs (env, defaults in brackets):
#   PORT [33500]  DEVICE [cuda]  BATCH_WAIT_MS [4]  MAX_BATCH [64]  STATE_TTL_S [900]
#   STEVE1_CKPT [CraftJarvis/MineStudio_STEVE-1.official]  MC_SIM_ENV [unset]  SIM_PY [python]
#   URL_FILE [unset; "http://<host>:<port>" is written there whenever /health answers]
#   MAX_RESTARTS [50]
set -o pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
[ -n "${MC_SIM_ENV:-}" ] && export PATH=$MC_SIM_ENV/bin:$PATH
SIM_PY=${SIM_PY:-python}
PORT=${PORT:-33500}
DEVICE=${DEVICE:-cuda}
BATCH_WAIT_MS=${BATCH_WAIT_MS:-4}
MAX_BATCH=${MAX_BATCH:-64}
STATE_TTL_S=${STATE_TTL_S:-900}
MAX_RESTARTS=${MAX_RESTARTS:-50}
export STEVE1_CKPT=${STEVE1_CKPT:-CraftJarvis/MineStudio_STEVE-1.official}
HOST=$(hostname -s)
echo "STEVE-1 service: http://${HOST}:${PORT} device=$DEVICE batch_wait_ms=$BATCH_WAIT_MS max_batch=$MAX_BATCH"

if [ -n "${URL_FILE:-}" ]; then
  mkdir -p "$(dirname "$URL_FILE")"; rm -f "$URL_FILE"
  ( while sleep 15; do
      if curl -sf --noproxy '*' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok": *true'; then
        [ -s "$URL_FILE" ] || { echo "http://${HOST}:${PORT}" > "$URL_FILE"; echo "[svc] READY -> $URL_FILE"; }
      else
        rm -f "$URL_FILE"
      fi
    done ) &
  trap 'kill $! 2>/dev/null; rm -f "$URL_FILE"' EXIT
fi

cd "$REPO/src/server/mc_env"
N=0
while :; do
  N=$((N + 1)); echo "[svc-wrap] launch #$N $(date +%F_%T)"
  "$SIM_PY" steve_service.py --port "$PORT" --device "$DEVICE" --batch-wait-ms "$BATCH_WAIT_MS" \
      --max-batch "$MAX_BATCH" --state-ttl-s "$STATE_TTL_S"
  echo "[svc-wrap] exit rc=$?; restart in 15 s"
  [ "$N" -ge "$MAX_RESTARTS" ] && { echo "[svc-wrap] giving up"; exit 1; }
  sleep 15
done
