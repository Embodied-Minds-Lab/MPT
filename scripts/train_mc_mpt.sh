#!/bin/bash
# train_mc_mpt.sh -- MPT on Minecraft (iron pickaxe, 120 turns), three agents (Logger / Miner / Smith).
#
# The MPT recipe:
#   * every turn the three agents bid in a sealed-bid auction on the levels {0, .2, .4, .6, .8}
#     (0 = pass; if every agent passes, the auction reopens at the 0.2 floor); the highest bidder
#     acts on its own action slice (parser-enforced verb whitelist);
#   * revenue-share settlement: the winning bid is the share of the turn's income passed back to
#     the previous winner; each winner's profit (minus rent 0.05) is its training reward
#     (profit_as_reward); losing bidders get 0;
#   * adv_estimator=reinforce_mean with a per-agent baseline (algorithm.group_by_agent_id=True);
#   * RL alternates 4 act-only iterations with 1 bid-only iteration. The bid iteration trains the
#     counterfactual answer-preference loss (cf_preference=True): on up to 4 contested
#     turns per episode the env is snapshotted before the winner acts; the winning agent's bid answer
#     is resampled (up to 6 times) until it would lose to the runner-up; the snapshot is restored
#     onto the two spare slots, the winner acts on one and the runner-up on the other, and each
#     branch runs 4 more turns. The winning agent's original and resampled bid answers form a
#     pair, preferred by branch reward, trained with a sequence-level Bradley-Terry loss.
#     Bid iterations carry no policy-gradient rows.
#   * start point: the format-SFT agents Xkev/mc_society_sft ({logger,miner,smith}/).
#
# The alternation phase lives in the file $MC_ALT_COUNTER (read-and-incremented once per training
# iteration), so a restart continues the 4:1 cycle where it stopped. The default path is per-EXP.
# Before launching, the counter is re-synchronised with the checkpoint the trainer will resume from
# (trainer.resume_mode=auto): it is set to the step in $CKPT_DIR/$EXP/latest_checkpointed_iteration.txt
# (N completed iterations -> counter N), or to 0 when there is no checkpoint. Without this, a crash
# between two saves would leave the counter up to SAVE_FREQ-1 iterations ahead of the resumed
# checkpoint (the 4:1 ratio would hold, but the cycle would start at a shifted position).
# MC_ALT_SYNC=0 disables the re-sync and uses the counter file as it is.
#
# Env-server requirements (scripts/start_mc_env.sh does all of this):
#   * exactly ONE server in MC_ENV_URL (the counterfactual collector drives one server's spare slots);
#   * started with MC_SNAPSHOT=1 (snapshot / restore / step_slot endpoints);
#   * --slots >= TBS*GROUP + 2 (the mainline batch plus two spare branch slots).
#
# algorithm.gamma=1.0 is passed explicitly: discounting is done by the revenue-share settlement, not
# by the advantage estimator.
#
# Usage:
#   MC_ENV_URL=http://<host>:31500 MC_SFT_DIR=/path/to/mc_society_sft bash scripts/train_mc_mpt.sh [hydra overrides...]
# Knobs (env, defaults in brackets):
#   MC_SFT_DIR [unset -> snapshot_download of MC_SFT_REPO=Xkev/mc_society_sft into $HF_HOME]
#   M_LOGGER/M_MINER/M_SMITH [$MC_SFT_DIR/{logger,miner,smith}]  DATA [<repo>/data/minecraft]
#   NGPUS [4] TBS [8] GROUP [1] MAXSTEPS [120] EXP [mc_mpt] RENT [0.05] GPU_MEM [0.4]
#   SAVE_FREQ [5] KEEP_CKPT [2] LOGGER ['["console"]']
#   CKPT_DIR [$MPT_WORK/checkpoints/mc] MC_ALT_COUNTER [$MPT_WORK/rl/.mc_alt_counter_$EXP] MC_ALT_SYNC [1]
#   MC_CHAIN_TRACE [$MPT_WORK/rl/mc_chain_trace_$EXP.jsonl]  plus the common.sh variables.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

LOGGER=${LOGGER:-'["console"]'}
MC_SFT_REPO=${MC_SFT_REPO:-Xkev/mc_society_sft}
if [ -z "${MC_SFT_DIR:-}" ] && { [ -z "${M_LOGGER:-}" ] || [ -z "${M_MINER:-}" ] || [ -z "${M_SMITH:-}" ]; }; then
  if [ "${DRY_RUN:-0}" = "1" ]; then
    MC_SFT_DIR=/path/to/mc_society_sft
  else
    echo "MC_SFT_DIR is unset: downloading $MC_SFT_REPO with huggingface_hub (set HF_HOME / HF_TOKEN as needed)."
    echo "  (equivalently: huggingface-cli download $MC_SFT_REPO --local-dir /path/to/mc_society_sft)"
    MC_SFT_DIR=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MC_SFT_REPO'))")
  fi
fi
M_LOGGER=${M_LOGGER:-$MC_SFT_DIR/logger}
M_MINER=${M_MINER:-$MC_SFT_DIR/miner}
M_SMITH=${M_SMITH:-$MC_SFT_DIR/smith}
DATA=${DATA:-$REPO/data/minecraft}
NGPUS=${NGPUS:-4}
TBS=${TBS:-8}
GROUP=${GROUP:-1}
MAXSTEPS=${MAXSTEPS:-120}
EXP=${EXP:-mc_mpt}
RENT=${RENT:-0.05}
GPU_MEM=${GPU_MEM:-0.4}
SAVE_FREQ=${SAVE_FREQ:-5}
KEEP_CKPT=${KEEP_CKPT:-2}
CKPT_DIR=${CKPT_DIR:-$MPT_WORK/checkpoints/mc}

: "${MC_ENV_URL:?set MC_ENV_URL to the env server (e.g. http://host:31500; see scripts/start_mc_env.sh)}"
case "$MC_ENV_URL" in
  *,*) echo "MC_ENV_URL must name exactly ONE env server for MPT training (the counterfactual collector"
       echo "branches on that server's spare slots); got: $MC_ENV_URL"; exit 1;;
esac
export MC_ENV_URL

# The alternation counter is ALWAYS file-backed so that a restart does not reset the phase.
export MC_ALT_COUNTER=${MC_ALT_COUNTER:-${MPT_ALT_COUNTER:-$MPT_WORK/rl/.mc_alt_counter_$EXP}}
# The trainer reads MPT_ALT_COUNTER before MC_ALT_COUNTER: point both at the one file this
# script resets, so a stray MPT_ALT_COUNTER cannot leave the trainer on another counter.
export MPT_ALT_COUNTER="$MC_ALT_COUNTER"
export MC_CHAIN_TRACE=${MC_CHAIN_TRACE:-$MPT_WORK/rl/mc_chain_trace_$EXP.jsonl}
[ "${DRY_RUN:-0}" = "1" ] || mkdir -p "$(dirname "$MC_ALT_COUNTER")" "$(dirname "$MC_CHAIN_TRACE")" "$CKPT_DIR"

# Re-synchronise the alternation counter with the checkpoint that resume_mode=auto will load.
if [ "${DRY_RUN:-0}" != "1" ] && [ "${MC_ALT_SYNC:-1}" = "1" ]; then
  _latest="$CKPT_DIR/$EXP/latest_checkpointed_iteration.txt"
  _ckpt_step=0
  if [ -s "$_latest" ]; then
    _ckpt_step=$(tr -dc '0-9' < "$_latest")
    _ckpt_step=${_ckpt_step:-0}
  fi
  _cur=$(tr -dc '0-9' 2>/dev/null < "$MC_ALT_COUNTER" || true)
  _cur=${_cur:-0}
  if [ "$_cur" != "$_ckpt_step" ]; then
    echo "alternation counter: $MC_ALT_COUNTER = $_cur, checkpoint step = $_ckpt_step -> resetting the counter to $_ckpt_step"
    [ "$_ckpt_step" = "0" ] && [ "$_cur" != "0" ] && \
      echo "  (no checkpoint under $CKPT_DIR/$EXP: this is a fresh run and starts at the first act iteration)"
  fi
  echo "$_ckpt_step" > "$MC_ALT_COUNTER"
fi

[ -f "$DATA/train.parquet" ] || { echo "missing $DATA/train.parquet"; exit 1; }

if [ "${DRY_RUN:-0}" != "1" ]; then
  for d in "$M_LOGGER" "$M_MINER" "$M_SMITH"; do
    [ -f "$d/config.json" ] || { echo "no config.json in $d (download Xkev/mc_society_sft, see README)"; exit 1; }
  done
  # Bounded health wait: a restarting env server needs a few minutes to rebuild its JVMs.
  wait_healthy "$MC_ENV_URL" 40 15 || { echo "env server never became healthy at $MC_ENV_URL"; exit 1; }
  HEALTH=$(curl -sf --noproxy '*' --max-time 10 "$MC_ENV_URL/health")
  NEED=$((TBS * GROUP + 2))
  "$PY" - "$HEALTH" "$NEED" <<'EOF' || exit 1
import json, sys
h = json.loads(sys.argv[1]); need = int(sys.argv[2])
slots = int(h.get("slots", 0))
if slots < need:
    sys.exit(f"env server has {slots} slots; MPT needs TBS*GROUP+2 = {need} (mainline batch + 2 branch slots)")
snap = h.get("mc_snapshot")
if snap is None:
    print("WARNING: /health does not report mc_snapshot; the env server must run with MC_SNAPSHOT=1")
elif str(snap) != "1":
    sys.exit(f"env server runs with MC_SNAPSHOT={snap}; the counterfactual bid phase needs MC_SNAPSHOT=1")
print(f"env server ok: slots={slots} (need {need}) mc_snapshot={snap} frames_per_action={h.get('frames_per_action')}")
EOF

  # Bounded GPU-drain guard: a previous trainer that is still tearing down (sglang engines, ray
  # workers) would make the new engines OOM. Wait up to 180 s, then warn and continue.
  _gpu_procs=""
  for _t in $(seq 1 18); do
    _gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
    [ -z "$_gpu_procs" ] && break
    echo "waiting for GPUs to drain ($_t/18); still occupied (pid, used_memory):"
    echo "$_gpu_procs"
    sleep 10
  done
  if [ -n "$_gpu_procs" ]; then
    echo "WARN: GPUs still occupied after 180 s -- launching anyway (pid, used_memory):"
    echo "$_gpu_procs"
  fi
fi

echo "MPT Minecraft: exp=$EXP alt=4 act : 1 bid (CF preference) counter=$MC_ALT_COUNTER ($(cat "$MC_ALT_COUNTER" 2>/dev/null || echo 0) iterations so far)"

run_or_echo "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_mean \
    algorithm.gamma=1.0 \
    algorithm.group_by_agent_id=True \
    data.train_files="$DATA/train.parquet" \
    data.train_batch_size="$TBS" \
    data.max_prompt_length=3072 \
    data.max_response_length=320 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TBS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    env.env_name=minecraft_bare \
    env.max_steps="$MAXSTEPS" \
    env.rollout.n="$GROUP" \
    agent.multi_agent=True \
    'agent.agent_ids=["Logger Desk (chain)","Miner Desk (chain)","Smith Desk (chain)"]' \
    "agent.model_ids=[\"$M_LOGGER\",\"$M_MINER\",\"$M_SMITH\"]" \
    agent.model_sharing=False \
    agent.orchestra_type=minecraft_chain \
    +agent.mixed_modality=True \
    agent.orchestra.minecraft_chain.revenue_share=True \
    agent.orchestra.minecraft_chain.bid_pass=True \
    "agent.orchestra.minecraft_chain.bid_levels=[0.0,0.2,0.4,0.6,0.8]" \
    agent.orchestra.minecraft_chain.reward_rent="$RENT" \
    agent.orchestra.minecraft_chain.profit_as_reward=True \
    agent.orchestra.minecraft_chain.bid_max_tokens=96 \
    agent.orchestra.minecraft_chain.alt_act_iters=4 \
    agent.orchestra.minecraft_chain.alt_bid_iters=1 \
    agent.orchestra.minecraft_chain.cf_preference=True \
    actor_rollout_ref.actor.bid_pref_enable=True \
    actor_rollout_ref.actor.bid_pref_scope=answer \
    reward_model.reward_manager=episode \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.project_name=mpt-minecraft \
    trainer.experiment_name="$EXP" \
    trainer.logger="$LOGGER" \
    trainer.default_local_dir="$CKPT_DIR/$EXP" \
    trainer.resume_mode=auto \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="$KEEP_CKPT" \
    trainer.max_critic_ckpt_to_keep="$KEEP_CKPT" \
    trainer.total_epochs=100 \
    "$@"
