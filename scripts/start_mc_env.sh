#!/bin/bash
# start_mc_env.sh -- the Minecraft env server (MineStudio + STEVE-1) used by MPT training and eval.
#
# Runs src/server/mc_env/mc_env_server.py in a supervised restart loop under a headless X server.
# Every slot is one Minecraft instance (one JVM). STEVE-1 turns each text instruction into
# MC_FRAMES low-level frames; it is loaded in-process on this host's GPU unless MC_STEVE_URL points
# at a shared STEVE-1 service (scripts/start_steve_service.sh).
#
# Slots: training needs TBS*GROUP + 2 (the mainline batch plus two spare branch slots that the
# counterfactual collector drives through /snapshot, /restore and /step_slot); evaluation needs at
# least as many slots as the seeds of one block (BLOCK in eval_mc.sh).
#
# Requirements: the MineStudio env (minestudio 1.1.6, JDK 8, xvfb-run on PATH; see
# configs/mc_env.sh.example), MINESTUDIO_DIR on persistent storage.
#
# Usage:  MINESTUDIO_DIR=/path/to/minestudio MC_SIM_ENV=/path/to/conda/envs/minestudio bash scripts/start_mc_env.sh
# Knobs (env, defaults in brackets):
#   MC_PORT [31500]  MC_SLOTS [10 = TBS 8 + 2]  MC_FRAMES [80, the benchmark value]
#   MC_SIM_ENV [unset; prefix put first on PATH, and the default JAVA_HOME]  MINESTUDIO_DIR [required]
#   JAVA_HOME [$MC_SIM_ENV]  STEVE1_CKPT [CraftJarvis/MineStudio_STEVE-1.official]
#   MC_STEVE_URL [unset = load STEVE-1 locally]  MC_DEVICE [cuda]
#   URL_FILE [unset; "http://<host>:<port>" is written there once the server is healthy]
#   MC_DISPLAY [random 100-899; the X display number -- set distinct values when several
#               servers share a node]  MC_MALMO_BASE_PORT [10000]  MC_RESET_DEADLINE_S [600]
#   MC_MAX_RESTARTS [20]  SIM_PY [python of MC_SIM_ENV, else python]
# No `set -u`: conda-forge openjdk activation scripts reference unbound variables.
set -o pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

: "${MINESTUDIO_DIR:?set MINESTUDIO_DIR (MineStudio engine cache on persistent storage)}"
export MINESTUDIO_DIR
if [ -n "${MC_SIM_ENV:-}" ]; then
  export PATH=$MC_SIM_ENV/bin:$PATH
  export JAVA_HOME=${JAVA_HOME:-$MC_SIM_ENV}
fi
[ -n "${JAVA_HOME:-}" ] && export PATH=$JAVA_HOME/bin:$PATH
SIM_PY=${SIM_PY:-python}

MC_PORT=${MC_PORT:-31500}
MC_SLOTS=${MC_SLOTS:-10}
MC_FRAMES=${MC_FRAMES:-80}
MC_DEVICE=${MC_DEVICE:-cuda}
MC_DISPLAY=${MC_DISPLAY:-$((100 + RANDOM % 800))}
MC_MAX_RESTARTS=${MC_MAX_RESTARTS:-20}
export STEVE1_CKPT=${STEVE1_CKPT:-CraftJarvis/MineStudio_STEVE-1.official}
[ -n "${MC_STEVE_URL:-}" ] && export MC_STEVE_URL

export MC_SNAPSHOT=1            # snapshot / restore / step_slot endpoints (counterfactual branches)
export MC_SCRIPTED_CRAFT=1      # semantic crafting/smelting (the furnace only needs to be in the inventory)
export MC_HISTORY=6
export MC_SOCKTIME=300          # dense-tree biomes can need > 120 s of world generation on the first reset
export MC_RENDER=640x360
export MC_VIDEO_SLOTS=0         # no per-frame video dumps
export _JAVA_OPTIONS="-Xmx8G"
export MC_MALMO_BASE_PORT=${MC_MALMO_BASE_PORT:-10000}
export MC_RESET_DEADLINE_S=${MC_RESET_DEADLINE_S:-600}
if command -v vglrun >/dev/null; then export MINESTUDIO_GPU_RENDER=1; else echo "no vglrun -> software rendering"; fi
command -v xvfb-run >/dev/null || { echo "xvfb-run not found on PATH (see configs/mc_env.sh.example)"; exit 1; }

# check_engine() prompts on stdin when the engine is missing; make sure it exists first.
"$SIM_PY" -c "from minestudio.simulator.entry import check_engine; check_engine(skip_confirmation=True); print('engine ready')" || exit 1

HOST=$(hostname -s)
URL="http://${HOST}:${MC_PORT}"
echo "MC env server: $URL (slots=$MC_SLOTS frames=$MC_FRAMES display=:$MC_DISPLAY steve=${MC_STEVE_URL:-local $MC_DEVICE})"

if [ -n "${URL_FILE:-}" ]; then
  mkdir -p "$(dirname "$URL_FILE")"
  rm -f "$URL_FILE"
  # publish the URL only once /health answers
  ( for _t in $(seq 1 720); do
      if curl -sf --noproxy '*' --max-time 10 "http://127.0.0.1:${MC_PORT}/health" 2>/dev/null | grep -q '"ok": *true'; then
        echo "$URL" > "$URL_FILE.tmp" && mv -f "$URL_FILE.tmp" "$URL_FILE"; echo "[srv-wrap] healthy -> $URL_FILE"; exit 0
      fi
      sleep 10
    done ) &
fi

cd "$REPO/src/server/mc_env"
# Supervised loop: a crashed server is relaunched (each relaunch rebuilds the JVMs, ~25 s per slot);
# the trainer waits for /health and continues.
N=0
while :; do
  N=$((N + 1))
  echo "[srv-wrap] launch #$N $(date +%F_%T)"
  xvfb-run -n "$MC_DISPLAY" "$SIM_PY" mc_env_server.py --port "$MC_PORT" --slots "$MC_SLOTS" \
      --frames-per-action "$MC_FRAMES" --action-mode steve1 --device "$MC_DEVICE"
  RC=$?
  echo "[srv-wrap] server exit rc=$RC; restart in 15 s"
  [ "$N" -ge "$MC_MAX_RESTARTS" ] && { echo "[srv-wrap] $MC_MAX_RESTARTS restarts -- giving up"; exit 1; }
  sleep 15
done
