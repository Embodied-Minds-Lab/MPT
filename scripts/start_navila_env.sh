#!/bin/bash
# start_navila_env.sh -- the habitat env server for NaVILA robot navigation (VLN-CE R2R).
#
# Runs src/server/navila_env/navila_env_server.py. One split per server: R2R episode ids restart at
# 1 in every split and the server resolves a reset by episode id alone, so training uses a
# NAV_SPLIT=train server (NAVILA_ENV_URL) and evaluation a separate NAV_SPLIT=val_unseen server
# (NAVILA_ENV_URL_VAL).
#
# Requirements (not redistributed): a habitat-sim 0.1.7 env (headless, with NaVILA's habitat-lab /
# VLN-CE stack), the NaVILA repository's evaluation/ tree (NAVILA_EVAL_DIR; its habitat configs
# resolve data/ relative to it) with the Matterport3D scenes and the R2R_VLNCE_v1-3_preprocessed
# episodes under its data/ directory.
#
# Usage:  NAVILA_EVAL_DIR=/path/to/NaVILA/evaluation NAV_SPLIT=val_unseen NAV_PORT=31601 bash scripts/start_navila_env.sh
# Knobs (env, defaults in brackets):
#   NAVILA_EVAL_DIR [required]  NAV_SPLIT [train]  NAV_PORT [31600]  NAV_SLOTS [8]  NAV_GPUS [1]
#   NAV_SIM_ENV [unset; prefix put first on PATH and its lib/ on LD_LIBRARY_PATH]  SIM_PY [python]
#   NAVILA_SLACK [unset = the config's slack reward; the reported runs used 0]
#   NAVILA_SNAPSHOT [0; set 1 on the training server when training with the counterfactual bid phase]
#   NAV_EGL_FIX [0; 1 = force the NVIDIA EGL vendor ICD and the system glvnd libraries, for nodes
#                where glvnd would otherwise pick a software (mesa) EGL and render black frames]
#   URL_FILE [unset; "http://<host>:<port>" is written there at start]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

: "${NAVILA_EVAL_DIR:?set NAVILA_EVAL_DIR to the evaluation/ directory of the NaVILA repository}"
export NAVILA_EVAL_DIR
NAV_SLOTS=${NAV_SLOTS:-8}
NAV_SPLIT=${NAV_SPLIT:-train}
NAV_PORT=${NAV_PORT:-31600}
NAV_GPUS=${NAV_GPUS:-1}
SIM_PY=${SIM_PY:-python}
WORK=${MPT_WORK:-$REPO/work}

if [ -n "${NAV_SIM_ENV:-}" ]; then
  export PATH=$NAV_SIM_ENV/bin:$PATH
  # the env's libstdc++/libLLVM (numba, imported by habitat) must win over an older system copy
  export LD_LIBRARY_PATH="$NAV_SIM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ "${NAV_EGL_FIX:-0}" = "1" ]; then
  ICD=$WORK/navila_egl/nvidia_only_icd; SYSGL=$WORK/navila_egl/sysgl
  mkdir -p "$ICD" "$SYSGL"
  for j in /usr/share/glvnd/egl_vendor.d/*nvidia*.json; do
    [ -e "$j" ] && ln -sfn "$j" "$ICD/$(basename "$j")"
  done
  ls "$ICD"/*.json >/dev/null 2>&1 || echo "WARNING: no NVIDIA EGL ICD in /usr/share/glvnd/egl_vendor.d -- rendering may fall back to mesa"
  export __EGL_VENDOR_LIBRARY_DIRS=$ICD
  for l in libEGL.so.1 libGLdispatch.so.0 libOpenGL.so.0 libGL.so.1 libGLX.so.0; do
    [ -e /usr/lib64/$l ] && ln -sfn /usr/lib64/$l "$SYSGL/$l"
  done
  # the system glvnd stack must come FIRST (the conda env ships its own libEGL/libGL)
  export LD_LIBRARY_PATH="$SYSGL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PYTHONNOUSERSITE=1 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONUNBUFFERED=1
[ -n "${NAVILA_SLACK:-}" ] && export NAVILA_SLACK
[ -n "${NAVILA_SNAPSHOT:-}" ] && export NAVILA_SNAPSHOT

HOST=$(hostname -s)
echo "NaVILA env server: http://${HOST}:${NAV_PORT} (slots=$NAV_SLOTS gpus=$NAV_GPUS split=$NAV_SPLIT slack=${NAVILA_SLACK:-config} snapshot=${NAVILA_SNAPSHOT:-0})"
if [ -n "${URL_FILE:-}" ]; then
  mkdir -p "$(dirname "$URL_FILE")"
  echo "http://${HOST}:${NAV_PORT}" > "$URL_FILE"
fi

# the habitat configs resolve data/ relative to the eval dir
cd "$NAVILA_EVAL_DIR"
exec "$SIM_PY" "$REPO/src/server/navila_env/navila_env_server.py" \
    --port "$NAV_PORT" --slots "$NAV_SLOTS" --split "$NAV_SPLIT" --num-gpus "$NAV_GPUS"
