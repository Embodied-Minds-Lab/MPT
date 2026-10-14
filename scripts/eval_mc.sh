#!/bin/bash
# eval_mc.sh -- held-out Minecraft evaluation of a three-agent MPT society.
#
# Held-out set: the 32 unseen world seeds 90900-90931 (data/minecraft/val.parquet; no training world
# uses them). Every seed is played RUNS times (independent samples at the training temperature), and
# eval/aggregate.py reports per-seed metrics plus mean +- SE over seeds.
#
# Prerequisites (both keep running while this script runs):
#   1. scripts/serve_mc_agents.sh  -- sglang servers for logger/miner/smith, writes MODELS_JSON;
#   2. scripts/start_mc_env.sh    -- one or more env servers (MC_SLOTS >= BLOCK each).
# The seeds are split into blocks of BLOCK; each env server in MC_ENV_URL is one lane that plays its
# share of the blocks one after another; lanes run in parallel. Finished (run, seed) pairs are
# skipped on a rerun, so the script can simply be restarted after an interruption. An OUT directory
# belongs to one set of agent checkpoints: the client refuses to resume into results produced with
# other checkpoints, so evaluate each checkpoint under its own TAG (or OUT).
#
# Usage:  MC_ENV_URL=http://h1:31500,http://h2:31500 bash scripts/eval_mc.sh
# Knobs (env, defaults in brackets):
#   MC_ENV_URL [required; comma list]  MODELS_JSON [$MPT_WORK/mc_eval/models.json]
#   TAG [mc_eval]  OUT [$MPT_WORK/mc_eval/$TAG]  RUNS [3]  BLOCK [4]  TURNS [120]
#   SEEDS [90900..90931, comma list]  VAL_TEMP [1.0]  DATA [<repo>/data/minecraft]
#   MC_ENV_TIMEOUT [3600]  MC_CHAIN_TRACE [$OUT/trace.jsonl]
#   FORCE [0; 1 = re-run every (run, seed) even if a result exists]  plus common.sh.
#
# Metrics: episode_reward (the event-reward ladder of the iron task, max 140);
# info.goal_reached (the iron pickaxe was crafted: the last stage of the ladder paid out);
# success.success_rate (the trainer's success flag `won`: ANY reward was obtained in the episode,
# i.e. at least one log; it is NOT the iron-pickaxe rate); episode_length (turns, <= 120).
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

: "${MC_ENV_URL:?set MC_ENV_URL to one or more env servers (comma separated)}"
MODELS_JSON=${MODELS_JSON:-$MPT_WORK/mc_eval/models.json}
TAG=${TAG:-mc_eval}
OUT=${OUT:-$MPT_WORK/mc_eval/$TAG}
RUNS=${RUNS:-3}
BLOCK=${BLOCK:-4}
TURNS=${TURNS:-120}
VAL_TEMP=${VAL_TEMP:-1.0}
DATA=${DATA:-$REPO/data/minecraft}
[ -z "${SEEDS:-}" ] && SEEDS=$(seq -s, 90900 90931)
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"
export MC_ENV_TIMEOUT=${MC_ENV_TIMEOUT:-3600}
export MC_CHAIN_TRACE=${MC_CHAIN_TRACE:-$OUT/trace.jsonl}
export MPT_EVAL_DUMP=${MPT_EVAL_DUMP:-$OUT/episodes.jsonl}
[ "${DRY_RUN:-0}" = "1" ] || mkdir -p "$OUT/logs"

IFS=',' read -r -a ALL <<< "$SEEDS"
IFS=',' read -r -a LANES <<< "$MC_ENV_URL"
NBLK=$(( (${#ALL[@]} + BLOCK - 1) / BLOCK ))
echo "MC eval: ${#ALL[@]} seeds x $RUNS runs, $NBLK blocks of $BLOCK over ${#LANES[@]} lane(s) -> $OUT"

if [ "${DRY_RUN:-0}" != "1" ]; then
  [ -s "$MODELS_JSON" ] || { echo "missing $MODELS_JSON -- start scripts/serve_mc_agents.sh first"; exit 1; }
  wait_healthy "$MC_ENV_URL" 40 15 || { echo "env server(s) not healthy: $MC_ENV_URL"; exit 1; }
fi

lane() {  # lane LANE_INDEX URL
  local li=$1 url=$2 b blk
  for ((b = li; b < NBLK; b += ${#LANES[@]})); do
    blk=$(IFS=,; echo "${ALL[*]:$((b * BLOCK)):$BLOCK}")
    echo "[lane $li] block $b seeds=$blk env=$url"
    MC_ENV_URL=$url run_or_echo "$PY" -u eval/mc_eval_client.py --seeds "$blk" --runs "$RUNS" \
        --models-json "$MODELS_JSON" --out "$OUT" --data "$DATA" --turns "$TURNS" \
        --temperature "$VAL_TEMP" --tag "${TAG}_b$b" ${FORCE_FLAG} \
        >> "$OUT/logs/lane${li}.log" 2>&1 || echo "[lane $li] block $b exited rc=$? (see $OUT/logs/lane${li}.log)"
  done
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for li in "${!LANES[@]}"; do
    for ((b = li; b < NBLK; b += ${#LANES[@]})); do
      blk=$(IFS=,; echo "${ALL[*]:$((b * BLOCK)):$BLOCK}")
      MC_ENV_URL=${LANES[$li]} run_or_echo "$PY" -u eval/mc_eval_client.py --seeds "$blk" --runs "$RUNS" \
          --models-json "$MODELS_JSON" --out "$OUT" --data "$DATA" --turns "$TURNS" --temperature "$VAL_TEMP" --tag "${TAG}_b$b" ${FORCE_FLAG}
    done
  done
  exit 0
fi

PIDS=()
for li in "${!LANES[@]}"; do
  lane "$li" "${LANES[$li]}" &
  PIDS+=($!)
done
wait "${PIDS[@]}"

# Every line of the dump is one write() of < 4 KB under O_APPEND, so lanes can share the file.
"$PY" eval/aggregate.py --dump "$MPT_EVAL_DUMP" --key seed \
    --metrics episode_reward,info.goal_reached,success.success_rate,episode_length --expect-instances "${#ALL[@]}" --out "$OUT"
