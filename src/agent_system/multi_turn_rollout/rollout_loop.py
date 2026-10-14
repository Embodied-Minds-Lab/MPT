import json
import os
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, preprocess_batch
from agent_system.environments import EnvironmentManagerBase
from agent_system.agent import BaseOrchestra
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment

        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # success_rate = {}
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    # for key, value in success_rate.items():
                    #     data[key] = value
                    # pass
                    data['pass'] = success['success_rate'][bs]

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs: EnvironmentManagerBase,
                                effective_rollout_n: int = 1, is_train: bool = True):
        """Implemented by MultiAgentTrajectoryCollector (MPT is multi-agent only)."""
        raise NotImplementedError("MPT rolls out through MultiAgentTrajectoryCollector")

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Run the multi-turn rollout loop.

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (training uses env.rollout.n, validation env.rollout.val_n).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Determine the effective rollout_n based on training/validation mode
        if is_train:
            effective_rollout_n = self.config.env.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
        else:
            # For validation, use val_n for pass@k and avg@k computation
            val_rollout_n = getattr(self.config.env.rollout, 'val_n', None)
            if val_rollout_n is not None and val_rollout_n > 1:
                effective_rollout_n = val_rollout_n
                gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
            else:
                effective_rollout_n = 1

        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
            self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                effective_rollout_n=effective_rollout_n,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output

# =============================================================================
# Multi‑Agent collector orchestrating a *team* of agents
# =============================================================================
class MultiAgentTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector that *delegates* action generation to the MPT agent society
    (a :class:`BaseOrchestra`: minecraft_chain, navila_chain or openresearcher_chain)."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Any,
        wg_to_agents_mapping: Dict[str, List[Dict[str, str]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any] = None,
    ):
        super().__init__(config=config, tokenizer=tokenizers, processor=processors)

        agent_ids = config.agent.agent_ids
        model_ids = config.agent.model_ids
        orchestra_type = config.agent.orchestra_type
        print("agent_ids: ", agent_ids)
        print("orchestra_type: ", orchestra_type)

        agents_to_wg_mapping = {}
        for wg_id, agents in wg_to_agents_mapping.items():
            for a in agents:
                agent_id = a['agent_id']
                agents_to_wg_mapping[agent_id] = wg_id

        if orchestra_type == "minecraft_chain":
            from agent_system.agent.orchestra.minecraft import MinecraftChainOrchestra as orchestra
        elif orchestra_type == "navila_chain":
            from agent_system.agent.orchestra.navila import NavilaChainOrchestra as orchestra
        elif orchestra_type == "openresearcher_chain":
            from agent_system.agent.orchestra.openresearcher import OpenResearcherChainOrchestra as orchestra
        else:
            raise ValueError(f"Unknown orchestra_type {orchestra_type!r}; this release ships "
                             f"minecraft_chain | navila_chain | openresearcher_chain")

        self.multiagent_orchestra: BaseOrchestra = orchestra(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )

    # ------------------------------------------------------------------
    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        effective_rollout_n: int,
        is_train: bool = True,
    ):

        batch_size = len(gen_batch.batch)

        _env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', None)
        obs, infos = envs.reset(kwargs=_env_kwargs)
        self.multiagent_orchestra.reset()
        # chain orchestras: an episode's auction spans MANY run() calls, so the orchestra needs to
        # know at episode start whether this rollout is train or val (it labels the debug trace and
        # advances the act/bid alternation only on training batches).
        if hasattr(self.multiagent_orchestra, 'begin_episode'):
            self.multiagent_orchestra.begin_episode(is_train=is_train)
        # Counterfactual preference phase (chain orchestras with cf_preference=True): the
        # collector gathers branch points during the episode and supplies the preference pairs
        # that are this iteration's ONLY training rows. Active only when the orchestra raised
        # cf_active for this iteration (train, bid phase of the alternation).
        _cf = getattr(self.multiagent_orchestra, 'cf_collector', None)
        _cf_on = _cf is not None and is_train and getattr(self.multiagent_orchestra, 'cf_active', False)
        # dp_actor's answer-level preference branch requires the pair keys on EVERY row whenever
        # actor.bid_pref_enable is on, also when no collector exists in this process.
        try:
            _pref_on = bool(self.config.actor_rollout_ref.actor.get('bid_pref_enable', False))
        except Exception:
            _pref_on = False
        if _cf_on:
            _cf.begin_iteration()

        if effective_rollout_n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % effective_rollout_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        # The last info a sample produced WHILE STILL ACTIVE. finalize_episode rows are
        # appended after the loop, when `infos` holds the FINAL iteration's observation --
        # and a sample that finished at turn 10 of a 64-turn batch reports {'won': False}
        # with no measures from turn 11 on. success_evaluator reads the info stored beside
        # the last active row, so without this every early-stopping success is logged as a
        # failure.
        last_active_infos: list = [None for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            ###############################
            text_actions, multiagent_batch_buffer = self.multiagent_orchestra.run(
                gen_batch=gen_batch,
                env_obs=obs,
                actor_rollout_wgs=actor_rollout_wg,
                active_masks=active_masks,
                step=_step+1
            )
            # counterfactual branch points must be snapshotted HERE: after run() picked this
            # turn's winners, strictly before envs.step executes their actions anywhere
            # (the driver's snapshot_state contract)
            if _cf_on:
                _cf.observe_turn(envs, obs, step=_step + 1)
            next_obs, rewards, dones, infos = envs.step(text_actions)
            ###############################
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"

            rewards_np = torch_to_numpy(rewards)
            for data in multiagent_batch_buffer:
                agent_id, agent_batch = data['agent_id'], data['batch']
                agent_batch.non_tensor_batch['agent_id'] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch['uid'] = uid_batch
                agent_batch.non_tensor_batch['traj_uid'] = traj_uid
                agent_batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                agent_batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
                # When the counterfactual feature is configured, EVERY row must carry the pair
                # keys (non-members as -1/0): dp_actor's answer branch fail-loudly requires
                # them on every batch it sees, PG iterations included.
                if (_cf is not None or _pref_on) and 'pref_pair_id' not in agent_batch.batch.keys():
                    _nrow = agent_batch.batch.batch_size[0]
                    agent_batch.batch['pref_pair_id'] = torch.full((_nrow,), -1, dtype=torch.long)
                    agent_batch.batch['pref_pair_role'] = torch.zeros(_nrow, dtype=torch.long)
                    agent_batch.batch['pref_pair_weight'] = torch.zeros(_nrow, dtype=torch.float32)
                # Economic credit of the row = the settled credit stamped by the orchestra
                # (econ_is_final is 0 on every chain-orchestra row, so the reward term is inert).
                if 'econ_credit_internal' in agent_batch.non_tensor_batch:
                    internal = agent_batch.non_tensor_batch['econ_credit_internal'].astype(np.float32)
                    is_final = agent_batch.non_tensor_batch['econ_is_final'].astype(np.float32)
                    econ_credit = internal + rewards_np.astype(np.float32) * is_final
                    agent_batch.non_tensor_batch['econ_credit'] = np.array(econ_credit, dtype=object)
                agent_batch_list: list[dict] = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]['agent_active_mask']:
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(infos[i])
                        last_active_infos[i] = infos[i]

            # revenue-share settlement: hand the orchestra this step's env rewards (settlement
            # needs the full per-turn stream).
            _note = getattr(self.multiagent_orchestra, 'note_step_rewards', None)
            if _note is not None:
                _note(rewards_np)

            # Update done states
            is_done = np.logical_or(is_done, dones)

            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        # END-OF-EPISODE FLUSH (chain orchestras, whose economy spans env turns). The revenue-share
        # settlement needs the full per-turn reward stream, so every winner row is held until
        # here. finalize_episode returns the same {'agent_id','batch'} shape as
        # multiagent_batch_buffer, with econ_credit_internal already carrying the full credit and
        # econ_is_final=0, so the combination below reproduces it unchanged.
        _finalize = getattr(self.multiagent_orchestra, 'finalize_episode', None)
        if _finalize is not None:
            _final_rewards = np.asarray(episode_rewards, dtype=np.float32)
            for data in _finalize(env_obs=obs, episode_rewards=episode_rewards, is_train=is_train):
                agent_id, agent_batch = data['agent_id'], data['batch']
                agent_batch.non_tensor_batch['agent_id'] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch['uid'] = uid_batch
                agent_batch.non_tensor_batch['traj_uid'] = traj_uid
                agent_batch.non_tensor_batch['rewards'] = torch_to_numpy(_final_rewards, is_object=True)
                # these rows WERE produced while their sample was active (on its own final turn);
                # gather_rollout_data drops any row whose active_masks is False.
                agent_batch.non_tensor_batch['active_masks'] = torch_to_numpy(np.ones(batch_size, dtype=bool), is_object=True)
                # same pair-key stamping as the in-loop buffer (dp_actor fail-loud contract)
                if (_cf is not None or _pref_on) and 'pref_pair_id' not in agent_batch.batch.keys():
                    _nrow = agent_batch.batch.batch_size[0]
                    agent_batch.batch['pref_pair_id'] = torch.full((_nrow,), -1, dtype=torch.long)
                    agent_batch.batch['pref_pair_role'] = torch.zeros(_nrow, dtype=torch.long)
                    agent_batch.batch['pref_pair_weight'] = torch.zeros(_nrow, dtype=torch.float32)
                if 'econ_credit_internal' in agent_batch.non_tensor_batch:
                    internal = agent_batch.non_tensor_batch['econ_credit_internal'].astype(np.float32)
                    is_final = agent_batch.non_tensor_batch['econ_is_final'].astype(np.float32)
                    econ_credit = internal + _final_rewards * is_final
                    agent_batch.non_tensor_batch['econ_credit'] = np.array(econ_credit, dtype=object)
                agent_batch_list: list[dict] = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]['agent_active_mask']:
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(
                            last_active_infos[i] if last_active_infos[i] is not None
                            else infos[i])

        # Counterfactual preference phase: replay the stored branch points and inject the
        # resulting pair rows -- the iteration's only training rows (the engine queued no PG
        # rows this iteration). Both members of a pair are appended ADJACENTLY to the same
        # sample's list, so the flattened batch keeps them next to each other (balance_batch
        # is skipped for pair-bearing batches in ray_trainer).
        if _cf_on:
            for _s, _rows, _meta in _cf.build_pairs(env_manager=envs, gen_batch=gen_batch,
                                                    actor_rollout_wgs=actor_rollout_wg):
                for _r in _rows:
                    _r['uid'] = uid_batch[_s]
                    _r['traj_uid'] = traj_uid[_s]
                    total_batch_list[_s].append(_r)
                    total_infos[_s].append(last_active_infos[_s]
                                           if last_active_infos[_s] is not None else infos[_s])

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        # Per-episode evaluation dump (the per-instance metric source of the eval scripts).
        # Off unless MPT_EVAL_DUMP is set, and never written on training batches. One JSON line
        # per episode, appended with a single write() so several eval lanes can share one file.
        # "rep" is the index inside a val_n group: the val batch is repeated with
        # interleave=True, so samples g*val_n + r share an instance.
        _dump = os.environ.get("MPT_EVAL_DUMP", "")
        if _dump and not is_train:
            _run = int(os.environ.get("MPT_EVAL_RUN", "0"))

            def _scalar(v):
                return v.item() if isinstance(v, np.generic) else v

            for i in range(batch_size):
                ek = dict(_env_kwargs[i]) if _env_kwargs is not None and _env_kwargs[i] is not None else {}
                info = last_active_infos[i] or {}
                rec = {"run": _run, "rep": int(i % max(1, effective_rollout_n)), "sample": i,
                       "traj_uid": str(traj_uid[i]),
                       "instance": {k: _scalar(ek[k]) for k in ("seed", "episode_id", "scene", "qid", "data_source")
                                    if k in ek},
                       "episode_reward": float(episode_rewards[i]), "episode_length": int(episode_lengths[i]),
                       "success": {k: float(np.asarray(v)[i]) for k, v in success.items()
                                   if np.asarray(v).ndim == 1 and len(np.asarray(v)) == batch_size},
                       "info": {k: _scalar(v) for k, v in info.items()
                                if isinstance(_scalar(v), (bool, int, float, str)) and k not in ("text",)}}
                with open(_dump, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings