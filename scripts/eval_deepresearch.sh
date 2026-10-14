#!/bin/bash
# eval_deepresearch.sh -- held-out deep-research evaluation of a three-agent society
# (Searcher / Opener / Finder; gemma-4-E2B-it agents).
#
# Held-out set: 50 OpenResearcher questions (data/deepresearch/val.parquet), browsed in
# the OpenResearcher corpus through the BM25 search service. The evaluation is the trainer's own
# validation pass (trainer.val_only=True): the agents are loaded under FSDP and served by the
# trainer's embedded sglang engines, so GPUs are required. Every question is played RUNS times
# inside one validation pass (env.rollout.val_n=RUNS, sampled at VAL_TEMP; greedy decoding loops on
# these 2B agents), every final answer is graded 0/1 by an LLM judge, and eval/aggregate.py reports
# per-question metrics plus mean +- SE over questions.
#
# This domain uses its own trainer env (env B: transformers 5.x + sglang main for gemma-4; see
# requirements_deepresearch.txt): point MPT_CONDA_ENV (or PY) at it.
# Deep research trains with the same MPT code as Minecraft (README, 3. Train, "Robot Navigation
# and Deep Research"); this evaluation switches the counterfactual bid phase off
# (cf_preference=False) and the alternation knobs are inert in a validation pass.
#
# Prerequisites:
#   * the OpenResearcher search service (scripts/start_or_search_service.sh) -> OR_SERVICE_URL;
#   * an OpenAI-compatible judge endpoint -> JUDGE_API_BASE / JUDGE_API_KEY (never printed).
#
# Usage:  DR_MODEL_DIR=/path/to/deepresearch_society OR_SERVICE_URL=http://host:8000 \
#         JUDGE_API_BASE=https://... JUDGE_API_KEY=... bash scripts/eval_deepresearch.sh [hydra overrides...]
# Knobs (env, defaults in brackets):
#   DR_MODEL_DIR [required; searcher/ opener/ finder/]  OR_SERVICE_URL [required]
#   JUDGE_API_BASE, JUDGE_API_KEY [required]  JUDGE_MODEL [qwen3.5-flash]
#   OR_DATE [2026-09-15; the "Today's date" line -- must equal the SFT prompts]
#   VAL_FILE [<repo>/data/deepresearch/val.parquet]  RUNS [3]  MAXSTEPS [64]  VBS [64]
#   NGPUS [4]  GPU_UTIL [0.25]  VAL_TEMP [1.0]  TOOL_DIALECT [auto]  ELIDE_MODE [stub]
#   PROMPT_LEN [16384]  RESP_LEN [2048]  PPO_TOK [8192]  OFFLOAD [True]  ATTN_IMPL [sdpa]  TP [1]
#   TAG [dr_eval]  OUT [$MPT_WORK/dr_eval/$TAG]  TIKTOKEN_CACHE_DIR [optional]
#   FORCE [0; 1 = move an existing $OUT/episodes.jsonl aside and evaluate again]  plus common.sh.
#
# Metrics: success.success_rate (the judge marked the final answer correct; OpenResearcher's HLE
# grader template, JUDGE_MODEL at temperature 0), success.answered_rate (the episode ended with an
# answer), success.truncated_rate (a turn hit the response cap, so no answer), episode_length
# (turns, <= MAXSTEPS), success.judge_error_rate (the judge call or its parse failed; such an answer
# scores 0, so a non-zero rate means the accuracy is a lower bound). The trainer log also prints
# val/.../avg@k and pass@k over the RUNS samples.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

: "${DR_MODEL_DIR:?set DR_MODEL_DIR to a directory with searcher/ opener/ finder/ (e.g. Xkev/deepresearch_society)}"
: "${OR_SERVICE_URL:?set OR_SERVICE_URL to the OpenResearcher search service (scripts/start_or_search_service.sh)}"
if [ "${DRY_RUN:-0}" != "1" ]; then
  : "${JUDGE_API_BASE:?export JUDGE_API_BASE (OpenAI-compatible judge endpoint)}"
  : "${JUDGE_API_KEY:?export JUDGE_API_KEY (the judge API key; never store it in a file in this repo)}"
fi
export OR_SERVICE_URL
JUDGE_MODEL=${JUDGE_MODEL:-qwen3.5-flash}
OR_DATE=${OR_DATE:-2026-09-15}
VAL_FILE=${VAL_FILE:-$REPO/data/deepresearch/val.parquet}
RUNS=${RUNS:-3}
MAXSTEPS=${MAXSTEPS:-64}
VBS=${VBS:-64}
NGPUS=${NGPUS:-4}
GPU_UTIL=${GPU_UTIL:-0.25}
VAL_TEMP=${VAL_TEMP:-1.0}
TOOL_DIALECT=${TOOL_DIALECT:-auto}
ELIDE_MODE=${ELIDE_MODE:-stub}
PROMPT_LEN=${PROMPT_LEN:-16384}
RESP_LEN=${RESP_LEN:-2048}
PPO_TOK=${PPO_TOK:-8192}
OFFLOAD=${OFFLOAD:-True}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
TP=${TP:-1}
TAG=${TAG:-dr_eval}
OUT=${OUT:-$MPT_WORK/dr_eval/$TAG}
M_SEARCHER=$DR_MODEL_DIR/searcher
M_OPENER=$DR_MODEL_DIR/opener
M_FINDER=$DR_MODEL_DIR/finder
[ -n "${TIKTOKEN_CACHE_DIR:-}" ] && export TIKTOKEN_CACHE_DIR

# gemma-4 + sglang: release only the KV cache between rollouts (never the weights), and run every
# engine's async generate on its own event loop (several engines per process).
export SGLANG_RELEASE_KV_ONLY=${SGLANG_RELEASE_KV_ONLY:-1}
export SGLANG_USE_ENGINE_LOOP=${SGLANG_USE_ENGINE_LOOP:-1}
export MC_DESK_PARALLEL=${MC_DESK_PARALLEL:-0}
export OR_TRACE=${OR_TRACE:-$OUT/or_env.jsonl}
export OR_CHAIN_TRACE=${OR_CHAIN_TRACE:-$OUT/or_chain.jsonl}
export MPT_EVAL_DUMP=${MPT_EVAL_DUMP:-$OUT/episodes.jsonl}

if [ "${DRY_RUN:-0}" != "1" ]; then
  [ -f "$VAL_FILE" ] || { echo "missing $VAL_FILE"; exit 1; }
  for d in "$M_SEARCHER" "$M_OPENER" "$M_FINDER"; do
    [ -f "$d/config.json" ] || { echo "no config.json in $d"; exit 1; }
  done
  ok=0
  for _t in $(seq 1 60); do
    if curl -s --noproxy '*' --max-time 10 "$OR_SERVICE_URL/" 2>/dev/null | grep -q '"active_searcher"'; then ok=1; break; fi
    echo "waiting for the search service at $OR_SERVICE_URL ($_t/60)"; sleep 20
  done
  [ "$ok" = 1 ] || { echo "search service never became ready at $OR_SERVICE_URL"; exit 1; }
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

echo "Deep-research eval: 50 questions x $RUNS runs, models=$DR_MODEL_DIR judge=$JUDGE_MODEL -> $OUT"
run_or_echo "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_mean \
    algorithm.gamma=1.0 \
    algorithm.group_by_agent_id=True \
    data.train_files="$VAL_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=1 \
    data.val_batch_size="$VBS" \
    data.max_prompt_length="$PROMPT_LEN" \
    data.max_response_length="$RESP_LEN" \
    data.return_raw_chat=True \
    data.truncation=left \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.model.text_only_processor=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OFFLOAD" \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_TOK" \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.bid_pref_enable=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
    actor_rollout_ref.rollout.max_model_len=$((PROMPT_LEN + RESP_LEN)) \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_TOK" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_UTIL" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    env.env_name=openresearcher \
    env.max_steps="$MAXSTEPS" \
    env.rollout.n=1 \
    env.rollout.val_n="$RUNS" \
    env.openresearcher.service_url="$OR_SERVICE_URL" \
    env.openresearcher.val_corpus=train \
    env.openresearcher.judge.model="$JUDGE_MODEL" \
    env.openresearcher.date="$OR_DATE" \
    env.openresearcher.tool_dialect="$TOOL_DIALECT" \
    env.openresearcher.elide_mode="$ELIDE_MODE" \
    agent.multi_agent=True \
    'agent.agent_ids=["Searcher Desk (openresearcher)","Opener Desk (openresearcher)","Finder Desk (openresearcher)"]' \
    "agent.model_ids=[\"$M_SEARCHER\",\"$M_OPENER\",\"$M_FINDER\"]" \
    agent.model_sharing=False \
    agent.orchestra_type=openresearcher_chain \
    agent.orchestra.openresearcher_chain.revenue_share=True \
    agent.orchestra.openresearcher_chain.bid_pass=True \
    "agent.orchestra.openresearcher_chain.bid_levels=[0.0,0.2,0.4,0.6,0.8]" \
    agent.orchestra.openresearcher_chain.seg_scale=False \
    agent.orchestra.openresearcher_chain.reward_rent=0.05 \
    agent.orchestra.openresearcher_chain.profit_as_reward=True \
    agent.orchestra.openresearcher_chain.bid_max_tokens=128 \
    agent.orchestra.openresearcher_chain.alt_act_iters=4 \
    agent.orchestra.openresearcher_chain.alt_bid_iters=1 \
    agent.orchestra.openresearcher_chain.cf_preference=False \
    reward_model.reward_manager=episode \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.project_name=mpt-deepresearch-eval \
    trainer.experiment_name="$TAG" \
    'trainer.logger=["console"]' \
    trainer.default_local_dir="$OUT/ckpt_unused" \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    "$@"

[ "${DRY_RUN:-0}" = "1" ] && exit 0
"$PY" eval/aggregate.py --dump "$MPT_EVAL_DUMP" --key qid --run-field rep --expect-instances 50 \
    --metrics success.success_rate,success.answered_rate,success.truncated_rate,success.judge_error_rate,episode_length \
    --out "$OUT"
