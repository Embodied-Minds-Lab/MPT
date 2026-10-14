# Every environment variable the scripts in scripts/ read, with its default. Copy to env.sh,
# uncomment and fill in what you need, and `source env.sh` before running a script.

# ---- common (scripts/common.sh) -------------------------------------------------------------
# export MPT_CONDA_ENV=          # trainer env prefix
# export PY=                    # interpreter
# export MPT_WORK=              # checkpoints / traces / counters / eval outputs
# export HF_HOME=               # Hugging Face cache
# export SGL_ATTENTION_BACKEND=triton  SGL_MM_ATTENTION_BACKEND=sdpa  WANDB_MODE=offline
# export DRY_RUN=1              # train/eval scripts print their command instead of running it

# ---- Minecraft env server -----------------
# export MINESTUDIO_DIR=        # MineStudio engine cache
# export MC_SIM_ENV=            # MineStudio env prefix
# export MC_PORT=31500 MC_SLOTS=10 MC_FRAMES=80 MC_DEVICE=cuda MC_DISPLAY= URL_FILE=
# export STEVE1_CKPT=CraftJarvis/MineStudio_STEVE-1.official MC_STEVE_URL=
# export MC_MALMO_BASE_PORT=10000 MC_RESET_DEADLINE_S=600

# ---- Minecraft training --------------------------------------------
# export MC_ENV_URL=            # ONE env server, e.g. http://host:31500
# export MC_SFT_DIR=            # local copy of Xkev/mc_society_sft (logger/ miner/ smith/)
# export NGPUS=4 TBS=8 GROUP=1 MAXSTEPS=120 EXP=mc_mpt RENT=0.05 GPU_MEM=0.4
# export SAVE_FREQ=5 KEEP_CKPT=2
# export CKPT_DIR= MC_ALT_COUNTER= MC_ALT_SYNC=1 MC_CHAIN_TRACE=

# ---- read by the training code itself (any setting; the MC_* names are still accepted) ----
# export MPT_ALT_COUNTER=       # alternation counter file; train_mc_mpt.sh sets it to its MC_ALT_COUNTER file
# export MPT_ALT_SYNC=1         # 0 = do not re-sync the counter with latest_checkpointed_iteration.txt
# export MPT_DESK_PARALLEL=0    # 1 = the agents' LLM calls of one auction phase run concurrently
# export MPT_CF_DUMP= MPT_CF_MARGIN_DUMP=   # counterfactual pair / margin inspection dumps (JSONL)

# ---- Minecraft eval ---------------------------------
# export MC_MODEL_DIR=          # the agents to evaluate
# export BASE_PORT=33000 MEM_FRAC=0.30 GPU=0 MODELS_JSON=
# export RUNS=3 BLOCK=4 TURNS=120 SEEDS= VAL_TEMP=1.0 MC_ENV_TIMEOUT=3600 TAG=mc_eval OUT= FORCE=0

# ---- NaVILA -----------------------------------
# export NAVILA_EVAL_DIR=       # NaVILA repository's evaluation/
# export NAV_SIM_ENV=           # habitat-sim 0.1.7 env prefix
# export NAV_SPLIT=val_unseen NAV_PORT=31601 NAV_SLOTS=24 NAV_GPUS=1 NAVILA_SLACK=0 NAV_EGL_FIX=0
# export NAV_MODEL_DIR=         # turner/ mover/ stopper/
# export NAVILA_ENV_URL_VAL=    # the val_unseen env server
# export RUNS=3 VBS=8 MAXSTEPS=64 NGPUS=4 GPU_MEM=0.4 VAL_TEMP=1.0 DATA_FILE= TAG=navila_eval OUT= FORCE=0

# ---- deep research ----------------------------------
# export OR_WORK_ROOT=          # lucene-jars/, bm25/, tmp/
# export OR_SERVICE_ENV=        # search-service env prefix
# export JAVA_HOME=             # JDK 21 for the search service
# export OR_INDEX_DIR= LUCENE_EXTRA_DIR= OR_START_TIMEOUT_S=1200 OR_INDEX_THREADS=8 OR_PROVISION= OR_FORCE=
# export DR_MODEL_DIR=          # searcher/ opener/ finder/
# export OR_SERVICE_URL=        # http://host:8000
# export JUDGE_API_BASE=        # OpenAI-compatible judge endpoint
# export JUDGE_API_KEY=         # set in your shell; never commit it
# export JUDGE_MODEL=qwen3.5-flash OR_DATE=2026-09-15 RUNS=3 MAXSTEPS=64 VBS=64 NGPUS=4 GPU_UTIL=0.25
# export VAL_TEMP=1.0 TOOL_DIALECT=auto ELIDE_MODE=stub TAG=dr_eval OUT= FORCE=0
