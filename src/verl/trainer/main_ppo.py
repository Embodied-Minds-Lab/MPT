# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


def _mirror_driver_output():
    """VERL_DRIVER_LOG=<file>: append a line-flushed, timestamped copy of this actor's stdout/stderr to a file
    we own. Ray forwards an actor's stdout to the driver through its log monitor, and on some clusters that
    channel has silently dropped EVERY stdout line of the TaskRunner (flush=True driver lines never reached
    the job log while worker lines did), so the job log cannot tell a hang from a driver that is printing. Prints
    from the WorkerDict processes still go through Ray; this mirror covers the driver only."""
    path = os.environ.get("VERL_DRIVER_LOG", "")
    if not path:
        return
    import sys
    import time

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", buffering=1)

    class _Tee:
        def __init__(self, orig, tag):
            self._orig, self._tag, self._bol = orig, tag, True

        def write(self, s):
            self._orig.write(s)
            for chunk in s.splitlines(True):
                if self._bol:
                    f.write(time.strftime("%m-%d %H:%M:%S ") + self._tag + " ")
                f.write(chunk)
                self._bol = chunk.endswith(("\n", "\r"))
            return len(s)

        def flush(self):
            self._orig.flush()
            f.flush()

        def __getattr__(self, name):
            return getattr(self._orig, name)

    sys.stdout = _Tee(sys.stdout, "OUT")
    sys.stderr = _Tee(sys.stderr, "ERR")
    print(f"[main_ppo] driver output mirrored to {path}", flush=True)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        _mirror_driver_output()
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        assert config.agent.multi_agent
        from agent_system.agent.utils import build_wg_ids, normalize_agent_id, normalize_model_id
        tokenizers: dict = {}
        processors: dict = {}
        
        wg_to_agents_mapping = build_wg_ids(config)
        for wg_id, agents_list in wg_to_agents_mapping.items():
            # download the checkpoint from hdfs
            model_id = agents_list[0]['model_id']
            local_path = copy_to_local(model_id, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            # used for multimodal LLM, could be none. model.text_only_processor=True forces None: a text-only run
            # on a multimodal architecture (gemma-4) must not take the VLM branches of multi_turn_rollout/utils.py
            processor = None if config.actor_rollout_ref.model.get("text_only_processor", False) else hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
            tokenizers[wg_id] = tokenizer
            processors[wg_id] = processor

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            # MPT runs on the FSDP workers only (the megatron worker of upstream verl is not shipped)
            raise NotImplementedError(f"actor strategy {config.actor_rollout_ref.actor.strategy!r}: "
                                      f"this release supports fsdp / fsdp2 only")

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        # MPT rewards come from the environments (EpisodeRewardManager below); an upstream verl
        # reward model can still be attached through the FSDP RewardModelWorker.
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        # MPT: profit-as-reward is a TRAIN-time switch read from the ACTIVE orchestra block,
        # agent.orchestra.<agent.orchestra_type>.profit_as_reward; evaluation (trainer.val_only)
        # always grades the env reward (R). Without it the settled revenue-share profit stamped
        # into econ_credit would be computed, logged and then discarded -- a winning bid and a
        # losing bid at the same turn would train on the identical scalar.
        _profit_as_reward = profit_as_reward_enabled(config)
        # MPT trains with plain REINFORCE (per-turn return minus a per-agent mean baseline). The other
        # estimators of the vendored verl code (gae, grpo, rloo, remax, reinforce_plus_plus, ...) are
        # unmodified upstream and are not supported by this release.
        assert str(config.algorithm.adv_estimator) == 'reinforce_mean', \
            f"MPT supports only algorithm.adv_estimator=reinforce_mean (got {config.algorithm.adv_estimator!r})."
        reward_fn = reward_manager_cls(tokenizers=tokenizers, num_examine=0, normalize_by_length=False,
                                       profit_as_reward=_profit_as_reward)

        # Evaluation mode (trainer.val_only) grades with the function-based RM; training runs never validate.
        val_only = bool(config.trainer.get("val_only", False))
        val_reward_fn = (reward_manager_cls(tokenizers=tokenizers, num_examine=1, normalize_by_length=False,
                                            profit_as_reward=False) if val_only else None)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import MultiAgentTrajectoryCollector

        traj_collector = MultiAgentTrajectoryCollector(config=config, wg_to_agents_mapping=wg_to_agents_mapping, tokenizers=tokenizers, processors=processors)

        from verl.utils.dataset.rl_dataset import collate_fn

        default_wg = list(wg_to_agents_mapping.keys())[0] # default model is the first one in model_ids
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizers[default_wg], processors[default_wg])
        val_dataset = (create_rl_dataset(config.data.val_files, config.data, tokenizers[default_wg], processors[default_wg])
                       if val_only else None)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizers=tokenizers,
            processors=processors,
            default_wg=default_wg,
            wg_to_agents_mapping=wg_to_agents_mapping,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def profit_as_reward_enabled(config) -> bool:
    """agent.orchestra.<agent.orchestra_type>.profit_as_reward of the ACTIVE chain orchestra (the
    other domains' blocks are ignored); False when the block or the key is missing."""
    _orch = config.agent.get('orchestra', None)
    _name = config.agent.get('orchestra_type', None)
    _blk = _orch.get(_name, None) if (_orch is not None and _name) else None
    return bool(_blk.get('profit_as_reward', False)) if _blk is not None else False


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
