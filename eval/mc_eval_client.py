import argparse, base64, io, json, os, socket, sys, time
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

DESKS = ["logger", "miner", "smith"]
AGENT_IDS = ["Logger Desk (chain)", "Miner Desk (chain)", "Smith Desk (chain)"]


# ----------------------------------------------------------------------------------------------
# the substituted piece: a worker group whose generate_sequences goes to an sglang HTTP server
# ----------------------------------------------------------------------------------------------
class RemoteSglang:
    """Mimics the subset of verl's ActorRollout worker group that the agents use: `world_size` and
    `generate_sequences(DataProto) -> DataProto`. Input/output layout copied from
    verl/workers/rollout/sglang_rollout/sglang_rollout.py (generate_sequences), so the batch that
    comes back unions with the agent's batch exactly as in training."""
    world_size = 1

    def __init__(self, url, tokenizer, rollout_cfg, label: str):
        # one URL or a list of replicas serving the same weights (models JSON "url" + "replicas").
        # Each request goes to the replica with the lowest recent latency (EWMA of the wall time of
        # this instance's own calls), with every 8th call round-robin so a replica that got slow is
        # re-probed; a retry moves on to the next replica. Lanes that started before a replica was
        # published stay on the primary, so later lanes drift to the less loaded card by themselves.
        self.urls = [u.rstrip("/") for u in ([url] if isinstance(url, str) else list(url))]
        self.url = self.urls[0]
        self._rr = 0
        self._lat = [0.0] * len(self.urls)          # EWMA seconds per call, 0 = never used
        self.tokenizer = tokenizer
        self.label = label
        self.response_length = int(rollout_cfg.response_length)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # the trainer's _init_sampling_params: fixed keys + every rollout-config key sglang knows
        self.base_params = dict(n=1, max_new_tokens=self.response_length, presence_penalty=0.0,
                                frequency_penalty=0.0, repetition_penalty=1.0, stop=None,
                                temperature=float(rollout_cfg.temperature),
                                top_k=int(rollout_cfg.top_k), top_p=float(rollout_cfg.top_p),
                                ignore_eos=bool(rollout_cfg.get("ignore_eos", False)),
                                skip_special_tokens=True)
        self.calls = 0
        self.tokens_out = 0
        self.seconds = 0.0

    @staticmethod
    def _img_b64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")            # lossless: the server decodes the same pixels
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _post(self, payload: Dict[str, Any], timeout: float = 1800.0, retries: int = 6):
        last = None
        for k in range(retries):
            n = len(self.urls)
            if n == 1 or k > 0 or self._rr % 8 == 0 or 0.0 in self._lat:
                i = (self._rr + k) % n                  # round-robin probe / retry on the next one
            else:
                i = min(range(n), key=lambda j: self._lat[j])
            self._rr += 1
            url = self.urls[i]
            try:
                t0 = time.time()
                r = requests.post(f"{url}/generate", json=payload, timeout=timeout)
                r.raise_for_status()
                dt = time.time() - t0
                self._lat[i] = dt if self._lat[i] == 0.0 else 0.7 * self._lat[i] + 0.3 * dt
                return r.json()
            except Exception as e:                      # noqa: BLE001 - the server may restart
                self._lat[i] = max(self._lat[i], 60.0)  # a failing replica is treated as slow
                last = e
                print(f"[remote-sglang {self.label}] /generate retry {k + 1}/{retries}: {e}", flush=True)
                time.sleep(min(300, 20 * (k + 1)))
        raise RuntimeError(f"[remote-sglang {self.label}] /generate failed after {retries} tries: {last}")

    def generate_sequences(self, prompts):
        from tensordict import TensorDict
        from verl import DataProto
        from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        raw_prompt_ids = [list(map(int, x)) for x in non_tensor_batch.pop("raw_prompt_ids")]
        image_data = None
        if "multi_modal_data" in non_tensor_batch:
            mm = non_tensor_batch.pop("multi_modal_data")
            image_data = [[self._img_b64(im) for im in (m.get("image") or [])] if isinstance(m, dict) else []
                          for m in mm]
            if all(len(x) == 0 for x in image_data):
                image_data = None
            else:
                image_data = [x if x else None for x in image_data]

        params = dict(self.base_params)
        params.update(prompts.meta_info.get("sampling_overrides") or {})
        if params.get("stop") is None:
            params.pop("stop")
        payload = {"input_ids": raw_prompt_ids, "sampling_params": params, "return_logprob": True}
        if image_data is not None:
            payload["image_data"] = image_data
        t0 = time.time()
        out = self._post(payload)
        self.seconds += time.time() - t0
        self.calls += 1
        if isinstance(out, dict):
            out = [out]
        # the trainer reads the ids out of output_token_logprobs (sglang_rollout._post_process_outputs)
        resp_ids, resp_lp = [], []
        for o in out:
            trip = o["meta_info"]["output_token_logprobs"]
            if trip:
                lp, ids = zip(*[(t[0], t[1]) for t in trip])
            else:
                lp, ids = (), ()
            resp_ids.append(torch.tensor(ids, dtype=torch.long))
            resp_lp.append(torch.tensor(lp, dtype=torch.float32))
            self.tokens_out += len(ids)
        from torch.nn.utils.rnn import pad_sequence
        response = pad_sequence(resp_ids, batch_first=True, padding_value=self.pad_token_id)
        log_probs = pad_sequence(resp_lp, batch_first=True, padding_value=float(self.pad_token_id))
        if response.shape[1] < self.response_length:
            response = pad_sequence_to_length(response, self.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.response_length, self.pad_token_id)
        response = response[:, :self.response_length].to(idx.device)
        log_probs = log_probs[:, :self.response_length].to(idx.device)
        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:   # qwen-vl mrope [bs, 3, P]
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch = TensorDict({"prompts": idx, "responses": response, "input_ids": seq,
                            "rollout_log_probs": log_probs, "attention_mask": attention_mask,
                            "position_ids": position_ids}, batch_size=batch_size)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)



# ----------------------------------------------------------------------------------------------
def build_config(args, model_paths: Dict[str, str], n: int, shard_parquet: str):
    """The trainer's hydra config with the overrides of scripts/train_mc_mpt.sh, minus the
    training-only knobs. Everything the rollout reads (prompt/response lengths, auction, rent, bid
    tokens, env) is identical. The eval is rollout-only: no counterfactual collector is built and
    the alternation is off."""
    from hydra import compose, initialize_config_dir
    overrides = [
        "algorithm.adv_estimator=reinforce_mean", "algorithm.gamma=1.0",
        "algorithm.group_by_agent_id=True",
        f"data.train_files={shard_parquet}", f"data.val_files={shard_parquet}",
        f"data.train_batch_size={n}", f"data.val_batch_size={n}",
        "data.max_prompt_length=3072", f"data.max_response_length={args.act_tokens}",
        "data.return_raw_chat=True", "+data.dataloader_num_workers=0",
        "actor_rollout_ref.model.path=null", "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.rollout.name=sglang", "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        "env.env_name=minecraft_bare", f"env.max_steps={args.turns}",
        "env.rollout.n=1", "env.rollout.val_n=1",
        "agent.multi_agent=True",
        "agent.agent_ids=" + json.dumps(AGENT_IDS),
        "agent.model_ids=" + json.dumps([model_paths[d] for d in DESKS]),
        "agent.model_sharing=False", "agent.orchestra_type=minecraft_chain",
        "+agent.mixed_modality=True",
        "agent.orchestra.minecraft_chain.revenue_share=True",
        "agent.orchestra.minecraft_chain.bid_pass=True",
        "agent.orchestra.minecraft_chain.bid_levels=[0.0,0.2,0.4,0.6,0.8]",
        "agent.orchestra.minecraft_chain.reward_rent=0.05",
        "agent.orchestra.minecraft_chain.profit_as_reward=True",
        f"agent.orchestra.minecraft_chain.bid_max_tokens={args.bid_tokens}",
        "agent.orchestra.minecraft_chain.cf_preference=False",
        # evaluation trains nothing: do not keep the bid rows (and their image tensors) of every turn
        "agent.orchestra.minecraft_chain.train_bid_rows=False",
        "agent.orchestra.minecraft_chain.alt_act_iters=0",
        "agent.orchestra.minecraft_chain.alt_bid_iters=0",
        "actor_rollout_ref.actor.bid_pref_enable=False",
        "reward_model.reward_manager=episode",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1",
        "trainer.project_name=mpt-minecraft-eval", f"trainer.experiment_name={args.tag}",
        "trainer.logger=[console]",
        # evaluation mode: make_envs builds the validation env pool (val_envs) only when val_only is set
        "trainer.val_only=True",
    ]
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "src", "verl", "trainer", "config")):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def all_val_seeds(data_dir: str) -> List[int]:
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    return [int(e["seed"]) for e in df["env_kwargs"]]


def write_shard_parquet(data_dir: str, seeds: List[int], path: str):
    """One row per seed: the seed's own row of val.parquet when it has one (exactly the reset
    payload the validation env pool sends), otherwise row 0 (task text, reward ladder, biome,
    max_high_steps) with only the seed replaced."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    by_seed = {int(e["seed"]): i for i, e in enumerate(df["env_kwargs"])}
    rows = []
    for s in seeds:
        if s in by_seed:
            rows.append(df.iloc[by_seed[s]].copy())
        else:
            r = df.iloc[0].copy()
            ek = dict(r["env_kwargs"])
            ek["seed"] = int(s)
            r["env_kwargs"] = ek
            rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.to_parquet(path, index=False)
    chk = pd.read_parquet(path)
    got = [int(e["seed"]) for e in chk["env_kwargs"]]
    assert got == list(seeds), (got, seeds)
    return path


def wait_health(url: str, path: str, what: str, tries: int = 240, sleep: float = 15.0):
    for _ in range(tries):
        try:
            r = requests.get(f"{url}{path}", timeout=15)
            if r.ok:
                return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        except Exception:
            pass
        time.sleep(sleep)
    raise RuntimeError(f"{what} at {url} never became healthy")


def atomic_json(path: str, obj: Dict[str, Any]):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="", help="comma-separated world seeds (default: every seed of --data/val.parquet)")
    ap.add_argument("--models-json", required=True,
                    help="written by scripts/serve_mc_agents.sh: {agent: {url, path}, ...}")
    ap.add_argument("--out", required=True, help="results root (results/run<r>/seed_<s>.json)")
    ap.add_argument("--data", default=os.path.join(REPO, "data", "minecraft"))
    ap.add_argument("--runs", type=int, default=3, help="independent episodes per seed")
    ap.add_argument("--turns", type=int, default=120)
    ap.add_argument("--act-tokens", type=int, default=320)   # data.max_response_length of training
    ap.add_argument("--bid-tokens", type=int, default=96)    # minecraft_chain.bid_max_tokens
    ap.add_argument("--temperature", type=float, default=1.0, help="rollout sampling temperature (training: 1.0)")
    ap.add_argument("--tag", default="mc_eval")
    ap.add_argument("--force", action="store_true", help="re-run (run, seed) pairs that already have a result")
    args = ap.parse_args()

    seeds_all = [int(x) for x in args.seeds.split(",") if x.strip()] or all_val_seeds(args.data)
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("MPT_EVAL_DUMP", os.path.join(args.out, "episodes.jsonl"))
    print(f"[mc-eval] seeds={seeds_all} runs={args.runs} dump={os.environ['MPT_EVAL_DUMP']}", flush=True)

    models = json.load(open(args.models_json))
    # "url" plus optional "replicas" (extra servers of the same weights) -> list per agent
    urls = {d: [models[d]["url"]] + list(models[d].get("replicas", [])) for d in DESKS}
    paths = {d: models[d]["path"] for d in DESKS}
    # An --out directory belongs to ONE set of agent checkpoints: resuming into results of other
    # checkpoints would silently skip their seeds and report the old numbers.
    if not args.force:
        _stale = []
        for _root, _dirs, _files in os.walk(os.path.join(args.out, "results")):
            for _fn in _files:
                if _fn.startswith("seed_") and _fn.endswith(".json"):
                    try:
                        _prev = json.load(open(os.path.join(_root, _fn))).get("models")
                    except Exception:  # noqa: BLE001 - an unreadable result is simply re-run
                        continue
                    if _prev != paths:
                        _stale.append(os.path.join(_root, _fn))
        if _stale:
            sys.exit(f"[mc-eval] {len(_stale)} result(s) under {args.out}/results were produced with other "
                     f"checkpoints (e.g. {_stale[0]}); use a new --out / TAG for this checkpoint, or --force "
                     f"to re-run every (run, seed) here")

    for d in DESKS:
        print(f"[mc-eval] {d}: {urls[d][0]} -> {wait_health(urls[d][0], '/get_model_info', f'model server {d}')}", flush=True)
    env_url = os.environ["MC_ENV_URL"]
    env_health = wait_health(env_url, "/health", "env server")
    print(f"[mc-eval] env server {env_url}: {env_health}", flush=True)

    from agent_system.environments import make_envs
    from agent_system.agent.utils import build_wg_ids
    from agent_system.multi_turn_rollout import MultiAgentTrajectoryCollector
    from verl import DataProto
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from torch.utils.data import DataLoader

    tokenizers, processors, wgs = None, None, None
    work = os.path.join(args.out, "work")
    os.makedirs(work, exist_ok=True)

    for r in range(args.runs):
        os.environ["MPT_EVAL_RUN"] = str(r)
        res_dir = os.path.join(args.out, "results", f"run{r}")
        os.makedirs(res_dir, exist_ok=True)
        seeds = [s for s in seeds_all if args.force or not os.path.exists(os.path.join(res_dir, f"seed_{s}.json"))]
        print(f"[mc-eval] run {r}: todo={seeds}", flush=True)
        if not seeds:
            continue
        n = len(seeds)
        shard = write_shard_parquet(args.data, seeds, os.path.join(work, f"shard_run{r}_{os.getpid()}.parquet"))
        cfg = build_config(args, paths, n, shard)

        # --- main_ppo's setup: envs, tokenizers/processors per worker group, collector -------
        # (envs are sized by the batch, so they are rebuilt per run; tokenizers and the HTTP
        # worker groups are built once)
        _, val_envs = make_envs(cfg)
        wg_to_agents = build_wg_ids(cfg)
        if wgs is None:
            tokenizers, processors, wgs = {}, {}, {}
            for wg_id, agents_list in wg_to_agents.items():
                model_path = agents_list[0]["model_id"]
                desk = DESKS[AGENT_IDS.index(agents_list[0]["agent_id"])]
                tok = hf_tokenizer(model_path, trust_remote_code=False)
                tokenizers[wg_id] = tok
                processors[wg_id] = hf_processor(model_path, trust_remote_code=False, use_fast=True)
                wgs[wg_id] = RemoteSglang(urls[desk], tok, cfg.actor_rollout_ref.rollout, desk)
                print(f"[mc-eval] wg {wg_id}: {urls[desk]} ({paths[desk]})", flush=True)
        collector = MultiAgentTrajectoryCollector(config=cfg, wg_to_agents_mapping=wg_to_agents,
                                                  tokenizers=tokenizers, processors=processors)
        default_wg = list(wg_to_agents.keys())[0]
        ds = create_rl_dataset(cfg.data.val_files, cfg.data, tokenizers[default_wg], processors[default_wg])
        dl = DataLoader(ds, batch_size=n, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)
        test_batch = DataProto.from_single_dict(next(iter(dl)))
        assert len(test_batch) == n
        got_seeds = [int(e["seed"]) for e in test_batch.non_tensor_batch["env_kwargs"]]
        assert got_seeds == seeds, (got_seeds, seeds)
        non_tensor_keys = ["raw_prompt_ids", "data_source", "env_kwargs"]
        for k in ("multi_modal_data", "raw_prompt", "tools_kwargs"):
            if k in test_batch.non_tensor_batch:
                non_tensor_keys.append(k)
        gen_batch = test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                   non_tensor_batch_keys=non_tensor_keys)
        # the trainer's ROLLOUT sampling (do_sample=True); validate=False so nothing switches to the
        # greedy val_kwargs path -- the sampling parameters live in RemoteSglang.base_params
        gen_batch.meta_info = {"eos_token_id": tokenizers[default_wg].eos_token_id,
                               "pad_token_id": tokenizers[default_wg].pad_token_id,
                               "recompute_log_prob": False, "do_sample": True, "validate": False}

        manifest = {"run": r, "seeds": seeds, "n": n, "models": paths, "urls": urls,
                    "env_url": env_url, "env_health": env_health,
                    "trace": os.environ.get("MC_CHAIN_TRACE", ""), "dump": os.environ["MPT_EVAL_DUMP"],
                    "turns": args.turns, "act_tokens": args.act_tokens, "bid_tokens": args.bid_tokens,
                    "sampling": wgs[default_wg].base_params, "host": socket.gethostname(),
                    "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        atomic_json(os.path.join(work, f"manifest_run{r}_{os.getpid()}.json"), manifest)

        # Live per-seed progress: after every env step the cumulative reward and turn count of each
        # seed go to $OUT/progress/run<r>/seed_<s>.json, so an interrupted lane leaves the partial
        # episode readable (the final result still needs a full episode).
        prog_dir = os.path.join(args.out, "progress", f"run{r}")
        os.makedirs(prog_dir, exist_ok=True)
        _cum = np.zeros(n, dtype=np.float64); _turn = [0] * n; _done = [False] * n
        _orig_step = val_envs.step

        def _step_logged(actions, _orig_step=_orig_step, _cum=_cum, _turn=_turn, _done=_done,
                         prog_dir=prog_dir, seeds=seeds, r=r):
            res = _orig_step(actions)
            try:
                _, rewards, dones, _ = res
                rw = np.asarray(rewards, dtype=np.float64).reshape(-1)
                dn = np.asarray(dones).reshape(-1)
                for i, s in enumerate(seeds):
                    if _done[i]:
                        continue
                    _cum[i] += float(rw[i]); _turn[i] += 1; _done[i] = bool(dn[i])
                    atomic_json(os.path.join(prog_dir, f"seed_{s}.json"),
                                {"run": r, "seed": s, "turn": _turn[i], "cum_reward": float(_cum[i]),
                                 "done": _done[i], "pid": os.getpid(),
                                 "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
            except Exception as e:  # noqa: BLE001 - progress logging must never break the rollout
                print(f"[mc-eval] progress log failed: {e}", flush=True)
            return res

        val_envs.step = _step_logged
        t0 = time.time()
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid, _ = collector.vanilla_multi_turn_loop(
            gen_batch=gen_batch, actor_rollout_wg=wgs, envs=val_envs, effective_rollout_n=1, is_train=False)
        wall = time.time() - t0
        # `won` is the trainer's success flag: ANY reward in the episode (the iron-pickaxe flag is
        # info.goal_reached in the per-episode dump)
        won = success.get("minecraft_success_rate", success.get("success_rate", np.zeros(n)))
        for i, s in enumerate(seeds):
            rec = {"run": r, "seed": s, "sample": i, "models": paths,
                   "episode_reward": float(episode_rewards[i]), "episode_length": int(episode_lengths[i]),
                   "won": bool(np.asarray(won)[i]) if i < len(won) else None, "traj_uid": str(traj_uid[i]),
                   "rows": int(len(total_batch_list[i])), "wall_s": round(wall, 1),
                   "trace": manifest["trace"], "pid": os.getpid(),
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            atomic_json(os.path.join(res_dir, f"seed_{s}.json"), rec)
        gen = {k: {"calls": w.calls, "tokens": w.tokens_out, "seconds": round(w.seconds, 1)} for k, w in wgs.items()}
        print(f"[mc-eval] DONE run={r} n={n} wall={wall / 60:.1f} min rewards={[float(x) for x in episode_rewards]} "
              f"mean={float(np.mean(episode_rewards)):.2f} lengths={[int(x) for x in episode_lengths]} gen={json.dumps(gen)}",
              flush=True)


if __name__ == "__main__":
    main()
