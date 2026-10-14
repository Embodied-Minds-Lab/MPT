# Shared environment for every script in scripts/. Source it, do not execute it:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Variables read (all optional):
#   MPT_CONDA_ENV   conda/venv prefix of the trainer env. When set, $MPT_CONDA_ENV/bin goes first on
#                   PATH, PY defaults to $MPT_CONDA_ENV/bin/python, and the env's bundled
#                   nvidia/*/lib dirs plus $MPT_CONDA_ENV/lib are put on LD_LIBRARY_PATH (tensordict's
#                   _C.so needs GLIBCXX_3.4.29, which older system libstdc++ lacks; the conda-forge
#                   libstdcxx in the env supplies it).
#   PY              python interpreter (default: $MPT_CONDA_ENV/bin/python, else python3)
#   MPT_WORK        root for checkpoints, traces, phase counters, eval outputs and logs
#                   (default: <repo>/work). Point this at large scratch storage.
#   SGL_ATTENTION_BACKEND (triton), SGL_MM_ATTENTION_BACKEND (sdpa), WANDB_MODE (offline),
#   TORCH_CUDA_ARCH_LIST (passed through unchanged when set)
#   DRY_RUN=1       train/eval scripts print their final command instead of running it

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export REPO
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}

# Proxy variables on compute nodes break sglang's loopback traffic and the env-server HTTP calls.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export MPT_WORK=${MPT_WORK:-$REPO/work}

if [ -n "${MPT_CONDA_ENV:-}" ]; then
  export PATH=$MPT_CONDA_ENV/bin:$PATH
  export CUDA_HOME=${CUDA_HOME:-$MPT_CONDA_ENV}
  for _d in "$MPT_CONDA_ENV"/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$MPT_CONDA_ENV/lib"
  PY=${PY:-$MPT_CONDA_ENV/bin/python}
fi
PY=${PY:-python3}
export PY

# flashinfer's JIT and the fa3 vision kernels are not available on every GPU/arch combination;
# these two backends are what all reported runs used.
export SGL_ATTENTION_BACKEND=${SGL_ATTENTION_BACKEND:-triton}
export SGL_MM_ATTENTION_BACKEND=${SGL_MM_ATTENTION_BACKEND:-sdpa}
export WANDB_MODE=${WANDB_MODE:-offline}
[ -n "${TORCH_CUDA_ARCH_LIST:-}" ] && export TORCH_CUDA_ARCH_LIST

# wait_healthy URL_LIST TRIES SLEEP
#   Poll every comma-separated URL's /health until all answer {"ok": true}; return 1 on timeout.
wait_healthy() {
  local urls=$1 tries=${2:-40} pause=${3:-15} t u ok
  for t in $(seq 1 "$tries"); do
    ok=1
    for u in ${urls//,/ }; do
      curl -sf --noproxy '*' --max-time 10 "$u/health" 2>/dev/null | grep -q '"ok": *true' || { ok=0; break; }
    done
    [ "$ok" = 1 ] && return 0
    echo "waiting for $urls ($t/$tries)"
    sleep "$pause"
  done
  return 1
}

# run_or_echo CMD...  -- execute, or print the command when DRY_RUN=1
run_or_echo() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}
