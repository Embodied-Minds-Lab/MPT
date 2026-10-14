#!/bin/bash
# eval_navila.sh -- held-out robot-navigation evaluation (NaVILA, VLN-CE R2R) of a three-agent MPT
# society (Turner / Mover / Stopper).
#
# Held-out set: 88 R2R-CE val_unseen episodes (data/navila/val.parquet). The evaluation is the trainer's own validation pass
# (trainer.val_only=True): the 8B agents are loaded under FSDP and served by the trainer's embedded
# sglang engines, so GPUs are required. Every episode is played RUNS times inside one validation
# pass (env.rollout.val_n=RUNS, sampled at VAL_TEMP), and eval/aggregate.py reports per-episode
# metrics plus mean +- SE over episodes.
#
# Prerequisites:
#   * a habitat env server for the val_unseen split (scripts/start_navila_env.sh with
#     NAV_SPLIT=val_unseen NAVILA_SLACK=0) with at least VBS*RUNS slots;
#   * the NaVILA sglang shims installed once into the trainer env (scripts/install_navila_sglang_shims.py).
#
# Usage:  NAV_MODEL_DIR=/path/to/navila_society NAVILA_ENV_URL_VAL=http://host:31601 bash scripts/eval_navila.sh [hydra overrides...]
# Knobs (env, defaults in brackets):
#   NAV_MODEL_DIR [required; turner/ mover/ stopper/]  NAVILA_ENV_URL_VAL [required]
#   DATA_FILE [<repo>/data/navila/val.parquet]  RUNS [3]  VBS [8]  MAXSTEPS [64]  NGPUS [4]
#   GPU_MEM [0.4]  VAL_TEMP [1.0]  TAG [navila_eval]  OUT [$MPT_WORK/navila_eval/$TAG]
#   FORCE [0; 1 = move an existing $OUT/episodes.jsonl aside and evaluate again]  plus common.sh.
#
# Metrics: episode_reward (summed VLN-CE WaypointReward: slack + 1.0 * reduction of the geodesic
# distance to the goal + 2.5 * success, per decision; slack = NAVILA_SLACK of the server, 0 here),
# success.success_rate (the episode's success flag), and the habitat measures at the end of the
# episode reported by the server: info.success, info.spl, info.ndtw, info.distance_to_goal,
# info.oracle_success.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

: "${NAV_MODEL_DIR:?set NAV_MODEL_DIR to a directory with turner/ mover/ stopper/ (e.g. Xkev/navila_society)}"
: "${NAVILA_ENV_URL_VAL:?set NAVILA_ENV_URL_VAL to the val_unseen habitat env server}"
DATA_FILE=${DATA_FILE:-$REPO/data/navila/val.parquet}
RUNS=${RUNS:-3}
VBS=${VBS:-8}
MAXSTEPS=${MAXSTEPS:-64}
NGPUS=${NGPUS:-4}
GPU_MEM=${GPU_MEM:-0.4}
VAL_TEMP=${VAL_TEMP:-1.0}
TAG=${TAG:-navila_eval}
OUT=${OUT:-$MPT_WORK/navila_eval/$TAG}
M_TURNER=$NAV_MODEL_DIR/turner
M_MOVER=$NAV_MODEL_DIR/mover
M_STOPPER=$NAV_MODEL_DIR/stopper

# make_envs also builds a (never stepped) training env pool from NAVILA_ENV_URL; point it at the
# val server with a batch of 1 so no train-split server is needed.
export NAVILA_ENV_URL_VAL
export NAVILA_ENV_URL=$NAVILA_ENV_URL_VAL
export MPT_EVAL_DUMP=${MPT_EVAL_DUMP:-$OUT/episodes.jsonl}

if [ "${DRY_RUN:-0}" != "1" ]; then
  if ! "$PY" -c "import pandas as pd, sys; n = len(pd.read_parquet(sys.argv[1])); sys.exit(0 if n == 88 else 1)" "$DATA_FILE" 2>/dev/null; then
    echo "$DATA_FILE is missing or does not have 88 rows (it ships with the repository in data/navila/)."
    exit 2
  fi
  for d in "$M_TURNER" "$M_MOVER" "$M_STOPPER"; do
    [ -f "$d/config.json" ] || { echo "no config.json in $d"; exit 1; }
  done
  if ! "$PY" -c "import importlib.util, os, sys; s = importlib.util.find_spec('sglang'); sys.exit(0 if s and os.path.exists(os.path.join(os.path.dirname(s.origin), 'srt', 'models', 'navila.py')) else 1)"; then
    echo "The NaVILA sglang shims are not installed in this env. Run once:"
    echo "  $PY scripts/install_navila_sglang_shims.py --env \"\$(dirname \"\$(dirname \"$(command -v "$PY")\")\")\""
    exit 1
  fi
  wait_healthy "$NAVILA_ENV_URL_VAL" 40 15 || { echo "env server not healthy: $NAVILA_ENV_URL_VAL"; exit 1; }
  NEED=$((VBS * RUNS))
  "$PY" - "$(curl -sf --noproxy '*' --max-time 10 "$NAVILA_ENV_URL_VAL/health")" "$NEED" <<'EOF' || exit 1
import json, sys
h = json.loads(sys.argv[1]); need = int(sys.argv[2])
slots = h.get("slots")
if slots is not None and int(slots) < need:
    sys.exit(f"env server has {slots} slots; this eval needs VBS*RUNS = {need}")
print(f"env server ok: {h}")
EOF
  mkdir -p "$OUT"
fi

# One validation pass writes every (instance, rep) record of the dump. Records of an earlier pass in
# the same OUT would be mixed into this one (eval/aggregate.py keeps the LAST record per (instance,
# rep)), so refuse to start over an existing dump unless FORCE=1, which moves it aside first.
if [ "${DRY_RUN:-0}" != "1" ] && [ -s "$MPT_EVAL_DUMP" ]; then
  if [ "${FORCE:-0}" = "1" ]; then
    _old="$MPT_EVAL_DUMP.prev.$(date +%Y%m%d_%H%M%S)"
    mv "$MPT_EVAL_DUMP" "$_old"
    echo "FORCE=1: moved the previous dump to $_old"
  else
    echo "$MPT_EVAL_DUMP already exists (an earlier evaluation in this OUT)."
    echo "Use a new TAG / OUT, or FORCE=1 to move the old dump aside and evaluate again."
    exit 1
  fi
fi

echo "NaVILA eval: 88 episodes x $RUNS runs, models=$NAV_MODEL_DIR -> $OUT"
run_or_echo "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_mean \
    algorithm.gamma=1.0 \
    algorithm.group_by_agent_id=True \
    data.train_files="$DATA_FILE" \
    data.val_files="$DATA_FILE" \
    data.train_batch_size=1 \
    data.val_batch_size="$VBS" \
    data.max_prompt_length=2560 \
    data.max_response_length=96 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.model.external_lib=agent_system.models.navila_register \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMP" \
    env.env_name=navila \
    env.max_steps="$MAXSTEPS" \
    env.rollout.n=1 \
    env.rollout.val_n="$RUNS" \
    +env.num_frames=8 \
    agent.multi_agent=True \
    'agent.agent_ids=["Turner Desk (navila)","Mover Desk (navila)","Stopper Desk (navila)"]' \
    "agent.model_ids=[\"$M_TURNER\",\"$M_MOVER\",\"$M_STOPPER\"]" \
    agent.model_sharing=False \
    agent.orchestra_type=navila_chain \
    agent.orchestra.navila_chain.alt_act_iters=0 \
    agent.orchestra.navila_chain.alt_bid_iters=0 \
    agent.orchestra.navila_chain.cf_preference=False \
    actor_rollout_ref.actor.bid_pref_enable=False \
    +agent.mixed_modality=True \
    reward_model.reward_manager=episode \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.project_name=mpt-navila-eval \
    trainer.experiment_name="$TAG" \
    'trainer.logger=["console"]' \
    trainer.default_local_dir="$OUT/ckpt_unused" \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    "$@"

[ "${DRY_RUN:-0}" = "1" ] && exit 0
"$PY" eval/aggregate.py --dump "$MPT_EVAL_DUMP" --key episode_id --run-field rep --expect-instances 88 \
    --metrics episode_reward,success.success_rate,info.success,info.spl,info.ndtw,info.distance_to_goal,info.oracle_success \
    --out "$OUT"
