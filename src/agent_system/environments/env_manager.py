import json
import re
from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy

import time


def _blank_frame(size=(128, 128)):
    """Last-resort stand-in so a lost Minecraft frame never yields an image-less prompt."""
    from PIL import Image
    return Image.new("RGB", size, (0, 0, 0))


def _decode_b64_image(b64):
    """base64 JPEG/PNG -> PIL RGB image; None on anything unreadable."""
    if not b64:
        return None
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image
        return Image.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB")
    except Exception:
        return None


def cf_enabled(config) -> bool:
    """True when the active chain orchestra block trains with the counterfactual bid phase."""
    try:
        blk = config.agent.orchestra[config.agent.orchestra_type]
        return bool(blk.get("cf_preference", False))
    except Exception:
        return False


def _cf_timeout(config) -> float:
    try:
        return float(config.agent.orchestra[config.agent.orchestra_type].get("cf_http_timeout", 1200.0))
    except Exception:
        return 1200.0


class MinecraftEnvironmentManager(EnvironmentManagerBase):
    """Minecraft: MULTI-TURN and multimodal.

    Unlike the single-step QA envs, every step returns a NEW frame — the agent sees the world it just
    changed. The current screenshot travels out of each env's info dict into observations['image'];
    the text channel carries the task, the decision counter and what the last instruction achieved.
    """
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self._last_images: List[Any] = []
        # env_name must be minecraft_bare: the server render passes through untouched (the MPT agents).
        if "bare" not in str(config.env.env_name).lower():
            raise ValueError(f"env.env_name={config.env.env_name!r}: the MPT Minecraft agents need "
                             f"env.env_name=minecraft_bare")

    def _pack(self, obs, infos):
        # EVERY sample must carry an image. minerl occasionally loses a frame (socket timeout ->
        # "sending random observation"), which used to leave image=None for that env; the prompt then
        # had no <image>, that sample had no pixel_values, and the whole micro-batch died in
        # dp_actor._forward_micro_batch with KeyError: 'pixel_values'. Fall back to the env's previous
        # frame (or black) so the batch stays homogeneous.
        images = [info.pop("image", None) for info in infos]
        if len(self._last_images) != len(images):
            self._last_images = [None] * len(images)
        for i, im in enumerate(images):
            if im is None:
                images[i] = self._last_images[i] or _blank_frame()
                print(f"[minecraft] env {i}: missing frame, reusing last/blank", flush=True)
            else:
                self._last_images[i] = im
        return {
            "text": self.build_text_obs(obs),
            "image": images,
            "anchor": list(obs),
        }

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        return self._pack(obs, infos), infos

    def step(self, text_actions: List[str]):
        # multi-agent (asserted in main_ppo): the orchestra already extracted the winning agent's action
        actions = text_actions
        t0 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        print(f"MinecraftEnv step time: {time.time() - t0:.2f}s", flush=True)

        self.tasks = next_obs                      # text obs is already fully rendered per env
        next_observations = self._pack(next_obs, infos)
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    # The env text observation now CONTAINS MODEL OUTPUT (the rolling history echoes back the
    # agent's own instructions). It is then .format()ed into a template that carries a literal
    # <image>, and preprocess_fn loops `while '<image>' in prompt` while supplying exactly ONE
    # image -- so a single stray '<image>' in model text is enough to crash training with
    # "IndexError: index 1 is out of bounds for dimension 0 with size 1".
    # Strip every vision/template control token from anything the env hands back, and say so
    # loudly if it ever fires, because that is the only way to learn where it came from.
    _INJECT = re.compile(r"<image>|<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>|<\|placeholder\|>")

    def _sanitize(self, s: str) -> str:
        cleaned, n = self._INJECT.subn("[img]", s or "")
        if n:
            print(f"[minecraft] STRIPPED {n} vision token(s) from the env observation -- "
                  f"model text leaked a template token", flush=True)
        return cleaned

    def build_text_obs(self, text_obs: List[str]) -> List[str]:
        text_obs = [self._sanitize(t) for t in text_obs]
        out = []
        for i in range(len(text_obs)):
            # the server render passes through untouched (it already carries TASK/RULES/history/contract)
            out.append(MINECRAFT_BARE_TEMPLATE.format(task_description=text_obs[i]))
        return out

    def cf_payload_to_obs(self, base_obs, s, payload):
        """Counterfactual env adapter: fold one branch /step_slot payload (text + image_b64) into a
        bs-wide observation positioned at sample ``s``, through the mainline path (build_text_obs on
        the server text). Only position ``s`` is consumed (every branch generation call masks to
        it); the other positions keep the base observation so prompt building stays shape-happy."""
        raw = list(base_obs.get("anchor") or base_obs["text"])
        raw[s] = payload.get("text", "")
        text = self.build_text_obs(raw)
        images = list(base_obs.get("image") or [None] * len(raw))
        im = _decode_b64_image(payload.get("image_b64"))
        if im is not None:
            images[s] = im
        return {"text": text, "image": images, "anchor": raw}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]['active_masks']:
                info = total_infos[batch_idx][i]
                won = float(info.get('won', 0.0))
                success['success_rate'].append(won)
                success[f"{info.get('data_source','minecraft')}_success_rate"].append(won)
                return
        # Same zero-row fallback as the base class (base.py): a counterfactual preference
        # iteration queues no policy-gradient rows and only pair-bearing samples get pair
        # rows injected, so a sample can legitimately end with an empty row list. Without a
        # defined append the batch-size assert in success_evaluator fires -- and because the
        # phase counter advances per attempt while no checkpoint exists yet, the crash would
        # replay and re-crash forever. There is no info dict to read data_source from here,
        # so the per-source array uses this manager's default source.
        success['success_rate'].append(0.0)
        success['minecraft_success_rate'].append(0.0)


class NavilaEnvironmentManager(EnvironmentManagerBase):
    """NaVILA robot navigation: MULTI-TURN and multimodal, mirroring MinecraftEnvironmentManager.

    Difference from MC: each observation carries the full 8-frame video WINDOW the policy sees
    (info["frames"] = list of 8 PIL images) rather than a single frame. That list travels into
    observations["image"][i]; the text channel carries the R2R instruction, which the Navigator
    agent formats into NaVILA's question template (8 <image> markers). Reward/done/measures come
    straight from the habitat env-server (official WaypointReward + SUCCESS/SPL/nDTW)."""

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self._last_frames: List[Any] = []
        self._num_frames = int(getattr(config.env, "num_frames", 8))
        # Per-step visualisation record. OFF unless NAVILA_VIZ_DIR is set, and then bounded to the
        # first NAVILA_VIZ_SAMPLES samples of each batch, because the frames are the only thing in
        # this pipeline that is never persisted: they arrive as PIL images in info["frames"], are
        # consumed into pixel_values, and are gone. Everything else needed to read a trajectory
        # (bids, winner, price, verb, whitelist) is already in the chain trace and joins to this on
        # (iteration, sample, step).
        self._viz_dir = os.environ.get("NAVILA_VIZ_DIR", "")
        self._viz_n = int(os.environ.get("NAVILA_VIZ_SAMPLES", "2"))
        # The counter is per PROCESS, so a resumed run restarts it at 0 and walks straight over the
        # previous run's records: a resume from a mid-run checkpoint would overwrite the
        # step-1 frames, and because steps.jsonl is opened in append mode the old and new step
        # records were concatenated into one 128-line file. NAVILA_VIZ_RUN namespaces the output so
        # a restart cannot destroy what an earlier run wrote; it defaults to the resume point when
        # the launcher supplies one, and falls back to the pid.
        run = os.environ.get("NAVILA_VIZ_RUN", "")
        if not run:
            resume = os.environ.get("NAVILA_RESUME_FROM", "")
            run = f"from{int(resume):05d}" if resume.isdigit() else f"pid{os.getpid()}"
        self._viz_dir = os.path.join(self._viz_dir, run) if self._viz_dir else ""
        self._viz_iter = -1
        self._viz_step = 0

    def _viz_on_reset(self, obs, infos):
        if not self._viz_dir:
            return
        try:
            self._viz_iter += 1
            self._viz_step = 0
            for i in range(min(self._viz_n, len(infos))):
                d = os.path.join(self._viz_dir, f"iter{self._viz_iter:05d}_s{i}")
                os.makedirs(d, exist_ok=True)
                # truncate any step log left by a previous run in this same slot; the step writer
                # appends, so a stale file would silently concatenate two episodes
                open(os.path.join(d, "steps.jsonl"), "w").close()
                with open(os.path.join(d, "meta.json"), "w") as f:
                    json.dump({"iteration": self._viz_iter, "sample": i,
                               "episode_id": infos[i].get("episode_id"),
                               "data_source": infos[i].get("data_source"),
                               "instruction": obs[i] if i < len(obs) else ""},
                              f, ensure_ascii=False, indent=2)
        except Exception as e:                                             # noqa: BLE001
            print(f"[navila-viz] reset record failed ({e}); continuing", flush=True)

    def _viz_on_step(self, frames, actions, rewards, dones, infos):
        """One JPEG of the agent's current view plus one JSONL line per step per recorded sample."""
        if not self._viz_dir:
            return
        try:
            for i in range(min(self._viz_n, len(infos))):
                d = os.path.join(self._viz_dir, f"iter{self._viz_iter:05d}_s{i}")
                os.makedirs(d, exist_ok=True)
                win = frames[i] if i < len(frames) else None
                if win:                       # last frame of the 8-frame window = current view
                    win[-1].convert("RGB").save(
                        os.path.join(d, f"step{self._viz_step:03d}.jpg"), quality=80)
                rec = {"step": self._viz_step,
                       "action": (actions[i] if i < len(actions) else "")[:400],
                       "reward": float(rewards[i]) if i < len(rewards) else None,
                       "done": bool(dones[i]) if i < len(dones) else None}
                for k in ("distance_to_goal", "spl", "ndtw", "success", "oracle_success",
                          "path_length", "won", "is_action_valid"):
                    if k in infos[i] and infos[i][k] is not None:
                        rec[k] = infos[i][k]
                with open(os.path.join(d, "steps.jsonl"), "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._viz_step += 1
        except Exception as e:                                             # noqa: BLE001
            print(f"[navila-viz] step record failed ({e}); continuing", flush=True)

    def _pack(self, obs, infos):
        # EVERY sample must carry its full 8-frame window; a lost window would leave that sample
        # without pixel_values and kill the multimodal micro-batch. Fall back to the env's previous
        # window (or all-black) so the batch stays homogeneous.
        frames = [info.pop("frames", None) for info in infos]
        if len(self._last_frames) != len(frames):
            self._last_frames = [None] * len(frames)
        for i, fr in enumerate(frames):
            if not fr:
                frames[i] = self._last_frames[i] or [_blank_frame() for _ in range(self._num_frames)]
                print(f"[navila] env {i}: missing frame window, reusing last/blank", flush=True)
            else:
                self._last_frames[i] = fr
        return {
            "text": self.build_text_obs(obs),
            "image": frames,           # each element is a LIST of 8 PIL frames
            "anchor": list(obs),
        }

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self._viz_on_reset(obs, infos)
        return self._pack(obs, infos), infos

    def step(self, text_actions: List[str]):
        # multi-agent (asserted in main_ppo): the orchestra already extracted the winning agent's action
        actions = text_actions
        t0 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        print(f"NavilaEnv step time: {time.time() - t0:.2f}s", flush=True)
        self.tasks = next_obs
        next_observations = self._pack(next_obs, infos)
        # after _pack, so the frames are the post-fallback windows the policy actually saw
        self._viz_on_step(next_observations["image"], actions, rewards, dones, infos)
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: List[str]) -> List[str]:
        # The env text is the R2R instruction; the Navigator agent template wraps it into NaVILA's
        # question with the 8 <image> markers, so here we pass it through unchanged.
        return list(text_obs)

    def cf_payload_to_obs(self, base_obs, s, payload):
        """Counterfactual env adapter: fold one branch /step_slot payload into a bs-wide observation
        positioned at sample ``s``. observations["image"][s] is a LIST of 8 PIL frames (the video
        window the policy sees), so payload["frames_b64"] is decoded by the client's own decoder;
        the text goes through build_text_obs like a mainline observation. Only position ``s`` is
        consumed (every branch generation call masks to it)."""
        from agent_system.environments.env_package.navila.envs import _decode_frames
        raw = list(base_obs.get("anchor") or base_obs["text"])
        raw[s] = payload.get("text", "")
        text = self.build_text_obs(raw)
        images = list(base_obs.get("image") or [None] * len(raw))
        frames = _decode_frames(payload.get("frames_b64"))
        if frames is not None:
            images[s] = frames
        return {"text": text, "image": images, "anchor": raw}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]['active_masks']:
                info = total_infos[batch_idx][i]
                won = float(info.get('won', 0.0))
                success['success_rate'].append(won)
                src = info.get('data_source', 'navila')
                success[f"{src}_success_rate"].append(won)
                # navigation-specific metrics for logging, when the server reports them
                for k in ("spl", "ndtw", "oracle_success"):
                    if k in info and info[k] is not None:
                        success[f"{src}_{k}"].append(float(info[k]))
                return
        # zero active rows (a CF-preference iteration queues no PG rows for this sample, only
        # pair rows get injected) -> mirror base._process_batch: append a defined 0.0 so
        # len(success['success_rate']) == batch_size and success_evaluator's assert holds.
        # Metric-only (success_rate is dilated for such samples), no PG/gradient effect.
        success['success_rate'].append(0.0)


class OpenResearcherEnvironmentManager(EnvironmentManagerBase):
    """OpenResearcher deep research: MULTI-TURN, text-only, native tool calling.

    The observation is the conversation itself. Besides the standard 'text'/'image'/'anchor' keys
    (text = the question, kept so BaseOrchestra.initialize_context and the anchor machinery work
    unchanged), the dict carries:
      'messages' : per env, the FULL OpenResearcher message list (system, question, assistant and
                   tool turns so far); agents render it with the chat template + tools=
      'tools'    : the OpenResearcher tool schemas (json.loads(TOOL_CONTENT)), shared
    Reward is terminal-only (judge verdict on the answering turn); every other step is 0.
    """
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        from agent_system.environments.env_package.openresearcher import TOOLS
        self._tools = TOOLS
        self.tasks: List[str] = []

    def _pack(self, messages_list):
        return {
            "text": list(self.tasks),
            "image": None,
            "anchor": list(self.tasks),
            "messages": messages_list,
            "tools": self._tools,
        }

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        # the question is the first user turn of every conversation
        self.tasks = [next((m["content"] for m in msgs if m.get("role") == "user"), "") for msgs in obs]
        return self._pack(obs), infos

    def step(self, text_actions: List[str]):
        # multi-agent (asserted in main_ppo): the orchestra already extracted the winning agent's action
        actions = text_actions
        t0 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        print(f"OpenResearcherEnv step time: {time.time() - t0:.2f}s", flush=True)
        next_observations = self._pack(next_obs)
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: List[str]) -> List[str]:
        return list(text_obs)

    def cf_payload_to_obs(self, base_obs, s, payload):
        """Counterfactual env adapter: the branch observation is the base observation with sample
        ``s``'s conversation replaced by the branch env's message list. text / anchor (the question)
        and tools are unchanged, so the agents render the branch prompt through the same chat-template
        path as a mainline prompt."""
        msgs = list(base_obs["messages"])
        msgs[s] = payload["messages"]
        return {**base_obs, "messages": msgs}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]['active_masks']:
                info = total_infos[batch_idx][i]
                won = float(info.get('won', 0.0))
                truncated = bool(info.get('truncated', False))
                success['success_rate'].append(won)
                success[f"{info.get('data_source', 'openresearcher')}_success_rate"].append(won)
                success['judge_error_rate'].append(float(bool(info.get('judge_error', False))))
                # a turn cut off at the response cap ends the episode but is NOT an answer
                success['answered_rate'].append(float(bool(info.get('done', False)) and not truncated))
                success['truncated_rate'].append(float(truncated))
                return
        # zero-row fallback (see base.py): keep the batch-size assert satisfied
        success['success_rate'].append(0.0)
        success['openresearcher_success_rate'].append(0.0)
        success['judge_error_rate'].append(0.0)
        success['answered_rate'].append(0.0)
        success['truncated_rate'].append(0.0)

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        # The trainer keeps only success_rate (-> pass@k) and drops every other key, so the
        # judge-health numbers are printed here, once per rollout batch, and land in the log.
        success = super().success_evaluator(*args, **kwargs)
        n = len(success['success_rate'])
        infos = kwargs.get('total_infos') or []
        rounds = [float(next((it.get('n_rounds', 0) for it in reversed(ti) if it), 0)) for ti in infos if ti]
        print(f"[openresearcher] batch n={n} answered={int(success['answered_rate'].sum())} "
              f"correct={int(success['success_rate'].sum())} "
              f"truncated={int(success['truncated_rate'].sum())} "
              f"judge_error={int(success['judge_error_rate'].sum())} "
              f"avg_rounds={np.mean(rounds) if rounds else 0.0:.1f}", flush=True)
        return success


def make_envs(config):
    """
    Create enviroments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # The validation env pool is built only in evaluation mode (trainer.val_only, or the Minecraft
    # eval client); training runs never validate and get val_envs=None.
    val_only = bool(config.trainer.get("val_only", False))
    # Get validation rollout n for pass@k and avg@k computation
    val_group_n = getattr(config.env.rollout, 'val_n', 1)
    # Training with the counterfactual bid phase: the train pool needs two spare branch slots
    # (servers: checked now, at startup, instead of at the first bid iteration; deep research:
    # two spare in-process envs). Evaluation pools and evaluation runs are untouched.
    cf_on = (not val_only) and cf_enabled(config)

    if "minecraft" in config.env.env_name.lower():
        from agent_system.environments.env_package.minecraft import (
            build_minecraft_envs, minecraft_projection)
        _K = int(getattr(config.env, "frames_per_action", 40))
        _envs = build_minecraft_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                     group_n=group_n, is_train=True, frames_per_action=_K)
        projection_f = partial(minecraft_projection)
        if cf_on:
            _envs.cf_bind(timeout=_cf_timeout(config))
        envs = MinecraftEnvironmentManager(_envs, projection_f, config)
        val_envs = None
        if val_only:
            _val_envs = build_minecraft_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                             group_n=val_group_n, is_train=False, frames_per_action=_K)
            val_envs = MinecraftEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "navila" in config.env.env_name.lower():
        from agent_system.environments.env_package.navila import (
            build_navila_envs, navila_projection)
        _envs = build_navila_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                  group_n=group_n, is_train=True)
        projection_f = partial(navila_projection)
        if cf_on:
            _envs.cf_bind(timeout=_cf_timeout(config))
        envs = NavilaEnvironmentManager(_envs, projection_f, config)
        val_envs = None
        if val_only:
            _val_envs = build_navila_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                          group_n=val_group_n, is_train=False)
            val_envs = NavilaEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "openresearcher" in config.env.env_name.lower():
        from agent_system.environments.env_package.openresearcher import (
            build_openresearcher_envs, openresearcher_projection)
        _envs = build_openresearcher_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                          group_n=group_n, is_train=True, env_config=config.env,
                                          n_spare=2 if cf_on else 0)
        if cf_on:
            _envs.cf_bind(timeout=_cf_timeout(config))
        projection_f = partial(openresearcher_projection)
        envs = OpenResearcherEnvironmentManager(_envs, projection_f, config)
        val_envs = None
        if val_only:
            _val_envs = build_openresearcher_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                                  group_n=val_group_n, is_train=False, env_config=config.env)
            val_envs = OpenResearcherEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        raise ValueError(f"Unsupported env.env_name {config.env.env_name!r}; this release ships "
                         f"minecraft* | navila* | openresearcher")
