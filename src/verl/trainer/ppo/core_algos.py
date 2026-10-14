# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from verl import DataProto
from typing import Sequence, List, Dict, Any


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_reinforce_mean_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    group_by_agent_id: bool = False,
):
    """REINFORCE with a batch-mean baseline.

    Every row is one turn; its scalar credit is token_level_rewards.sum(-1) -- in MPT the
    settled per-turn profit minus rent (the economy already did temporal credit assignment).
        w_row = credit_row - mean(credits in the row's baseline group)
    Baseline group: the whole batch per agent when group_by_agent_id (each agent's trainer
    signal is baselined within that agent), else per prompt-group index. NO std
    normalization, NO trajectory averaging, NO clipping -- deliberately bare.
    """
    scores = token_level_rewards.sum(dim=-1)
    id2scores = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2scores[index[i]].append(scores[i])
        id2mean = {idx: torch.stack(v).mean() if len(v) > 1 else torch.tensor(0.0)
                   for idx, v in id2scores.items()}
        adv = scores.clone()
        for i in range(bsz):
            adv[i] = scores[i] - id2mean[index[i]].to(scores.dtype)
        adv = adv.unsqueeze(-1) * response_mask
    return adv, adv


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    print("group_by_agent_id: ", group_by_agent_id)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj_accumulator = defaultdict(list)
    traj2avg = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])
        
        for (idx, t_idx), reward_list in traj_accumulator.items():
            if group_by_agent_id:
                id2score[idx].extend(reward_list)
            else:
                avg_score = torch.stack(reward_list).mean()
                traj2avg[(idx, t_idx)] = avg_score
                id2score[idx].append(avg_score)
        if not group_by_agent_id:
            for i in range(bsz):
                scores[i] = traj2avg[(index[i], traj_index[i])]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_answer_preference_loss(log_prob, w_idx, l_idx, response_mask_w, response_mask_l,
                                   pair_weight=None, beta=1.0, margin_sat_threshold=20.0,
                                   old_log_prob=None, ref_anchor=False, return_margins=False):
    """Whole-answer pairwise preference loss (counterfactual branching protocol).

    The SAME agent samples two complete candidate bid answers for the SAME context (the
    original answer vs. the ranking-changing resample), each branch rolls out p further
    turns, and the state delta decides the winner. Both texts are real full generations with
    no shared suffix to cancel, so the comparison must cover the WHOLE sequence:

        m    = sum_t logpi(y_w[t]|.) * mask_w[t]  -  sum_t logpi(y_l[t]|.) * mask_l[t]
        loss = -log sigmoid(beta * m)

    This is reference-free sequence-level Bradley-Terry (DPO-shaped minus the ref term; the
    two answers share agent and context and are close in length, so the length bias is
    limited -- if length hacking is ever observed, add length normalization explicitly, not
    silently here).

    Reference anchor (ref_anchor=True with old_log_prob given):

        m_raw = lp_w - lp_l                                      (the unanchored margin above)
        m     = (lp_w - lp_w_old) - (lp_l - lp_l_old)

    where lp_*_old are the SAME masked sums taken over old_log_prob, the sampling-time
    log-probs of the same rows (dp_actor passes the micro-batch's "old_log_probs"). This is
    the DPO reference term with the sampling policy as the reference. Why the raw margin is
    not enough: both answers are samples from the SAME policy, so at beta=1 m_raw is
    dominated by the answers' unconditional likelihood/length difference, not by the bid
    content. Observed on the first real pair: logp_w=-56.34 (43 tokens), logp_l=-69.45
    (52 tokens), m_raw=+13.1 nats, loss=6e-06, gradient ~0 -- the pair contributed nothing
    because the winner happened to be the likelier sample; with the roles flipped it would
    have taken a fully saturated step, equally unrelated to the bid. With the anchor, the
    current policy and the sampling policy coincide at the first update, so m = 0 for EVERY
    pair (exactly, up to forward-pass numerics): loss = log 2 and
        dL/dtheta = -(beta/2) * weight * (grad logp(y_w) - grad logp(y_l)),
    i.e. the Bradley-Terry gradient is never saturated by the answers' unconditional
    likelihood/length difference -- only the change since sampling enters the margin. The
    anchor terms are constants (old_log_prob carries no graph), so a pair's gradient
    DIRECTION is unchanged; only its magnitude sigmoid(-beta*m) is re-based. With
    ref_anchor=False (the default) every line below computes exactly what it did before the
    argument existed.

    The margin is deliberately NOT clamped (diverging from the sibling's margin_clamp=8):
    -logsigmoid(beta*m) is numerically stable for any m (the deep-negative end is softplus's
    linear tail), and the gradient |dL/dm| = beta * sigmoid(-beta*m) <= beta is naturally
    bounded, so clamping guards an explosion that cannot happen -- while a hard clamp has an
    EXACTLY-zero derivative outside the window and would silently zero out the deeply
    mis-ordered pairs that need gradient the most. The gradient path therefore uses the bare
    logsigmoid; margin_sat_threshold is reporting-only (fraction of pairs with |m| above it).

    How pairs enter the batch: y_w and y_l both appear as ORDINARY rows in the same
    micro-batch (counterfactual rows are injected by the collector with advantage zeroed and
    serve only this loss; dp_actor drops them whole-row from PG/entropy/KL) -- no new batch
    structure is invented. Pairing is content-addressed: a pair's two rows share one
    "pref_pair_id", and "pref_pair_role" marks +1 winner / -1 loser. The caller (dp_actor's
    bid_pref_scope="answer" branch) matches w_idx/l_idx by id WITHIN the current micro-batch
    before calling here, so no reorder or split can point a pair at the wrong rows; a pair
    split across micro-batches is counted, warned about, and skipped by the caller. The
    bounds check below is the last line of defense: out-of-range indices raise, never pass
    silently.

    Args:
        log_prob:        (bs, resp_len) log-probabilities of the actually generated tokens
                         from the main forward pass
        w_idx / l_idx:   (n_pairs,) long, row indices of the winning / losing answers WITHIN
                         this micro-batch (produced by the caller's pref_pair_id matching)
        response_mask_w: (n_pairs, resp_len) the winner rows' response mask
                         (caller passes response_mask[w_idx])
        response_mask_l: (n_pairs, resp_len) the loser rows' response mask
        pair_weight:     (n_pairs,) per-pair weight (|branch score delta| is the suggested
                         normalization); None means all ones; 0 skips the pair
        beta:            temperature -- small beta is gentle, large approaches a hard
                         preference
        margin_sat_threshold: reporting-only threshold: valid pairs with |m| above it count
                         toward the returned sat_frac (monitoring the share of huge-margin
                         pairs); it never clamps and never touches the gradient (see above)
        old_log_prob:    (bs, resp_len) or None: sampling-time log-probs of the SAME rows as
                         log_prob (dp_actor passes data["old_log_probs"]). Read only when
                         ref_anchor is True; treated as a constant (detached).
        ref_anchor:      anchor the margin on old_log_prob (see the reference-anchor
                         paragraph). True with old_log_prob=None falls back to the unanchored
                         margin -- the caller is responsible for logging that fallback.
        return_margins:  if True, append a 4th return element: a dict of DETACHED per-pair
                         tensors (length n_pairs, same order as w_idx) for metrics / the
                         margin dump: "margins_raw" (lp_w - lp_l), "margins" (the margin
                         that entered the loss: anchored when ref_anchor took effect, else
                         identical to margins_raw), "lp_w", "lp_l", "lp_w_old", "lp_l_old"
                         (the latter two None when not anchored), "valid" (bool, weight > 0)
                         and "anchored" (bool). The default False keeps the 3-tuple return.

    Returns:
        (loss, n_pairs, sat_frac) -- loss is averaged over participating pairs; sat_frac is
        the fraction of valid pairs with |m| > margin_sat_threshold (float, monitoring only;
        m is the anchored margin when ref_anchor took effect). With no valid pairs, returns
        (0, 0, 0.0) with a graph-connected zero loss. With return_margins=True, (loss, n_pairs, sat_frac,
        margins_info) instead.
    """
    bs = log_prob.shape[0]
    n_pairs = w_idx.shape[0]
    anchored = bool(ref_anchor) and old_log_prob is not None

    def _empty_info():
        _e = torch.zeros(0, dtype=log_prob.dtype, device=log_prob.device)
        return {"margins_raw": _e, "margins": _e, "lp_w": _e, "lp_l": _e,
                "lp_w_old": _e if anchored else None, "lp_l_old": _e if anchored else None,
                "valid": torch.zeros(0, dtype=torch.bool, device=log_prob.device),
                "anchored": anchored}

    if n_pairs == 0:
        z = log_prob.sum() * 0.0          # keep the graph connected so DDP sees no unused params
        return (z, 0, 0.0, _empty_info()) if return_margins else (z, 0, 0.0)
    if int(w_idx.max()) >= bs or int(l_idx.max()) >= bs or int(w_idx.min()) < 0 or int(l_idx.min()) < 0:
        raise ValueError(
            f"compute_answer_preference_loss: pair indices out of range (bs={bs}, "
            f"w_idx in [{int(w_idx.min())},{int(w_idx.max())}], l_idx in [{int(l_idx.min())},{int(l_idx.max())}]). "
            "The caller must match local indices by pref_pair_id within THIS micro-batch; "
            "out-of-range means the matching code or the injection encoding is broken -- "
            "fix upstream, do not swallow it here."
        )
    if pair_weight is None:
        pair_weight = torch.ones(n_pairs, dtype=log_prob.dtype, device=log_prob.device)
    valid = pair_weight > 0
    lp_w = (log_prob[w_idx] * response_mask_w).sum(dim=-1)   # (n_pairs,) whole-answer logpi, winner
    lp_l = (log_prob[l_idx] * response_mask_l).sum(dim=-1)   # (n_pairs,) whole-answer logpi, loser
    m_raw = lp_w - lp_l                                      # unclamped: see the margin paragraph
    lp_w_old = lp_l_old = None
    if anchored:
        # Reference anchor: the same masked sums over the sampling-time log-probs of the
        # same rows. old_log_prob is a constant here (no graph), so the anchor only re-bases
        # the margin; the gradient direction of each pair is unchanged.
        if tuple(old_log_prob.shape) != tuple(log_prob.shape):
            raise ValueError(
                f"compute_answer_preference_loss: old_log_prob shape {tuple(old_log_prob.shape)} "
                f"!= log_prob shape {tuple(log_prob.shape)}; the anchor must be the sampling-time "
                "log-probs of the SAME micro-batch rows."
            )
        _old = old_log_prob.detach().to(log_prob.dtype)
        lp_w_old = (_old[w_idx] * response_mask_w).sum(dim=-1)
        lp_l_old = (_old[l_idx] * response_mask_l).sum(dim=-1)
        m = (lp_w - lp_w_old) - (lp_l - lp_l_old)
    else:
        m = m_raw
    # Per-pair inspection tensors (detached, length n_pairs, w_idx order); populated even
    # when every pair is weighted out so the caller's dump keeps full per-pair columns.
    info = {"margins_raw": m_raw.detach(), "margins": m.detach(),
            "lp_w": lp_w.detach(), "lp_l": lp_l.detach(),
            "lp_w_old": lp_w_old.detach() if lp_w_old is not None else None,
            "lp_l_old": lp_l_old.detach() if lp_l_old is not None else None,
            "valid": valid.detach(), "anchored": anchored}
    if not bool(valid.any()):
        z = log_prob.sum() * 0.0
        return (z, 0, 0.0, info) if return_margins else (z, 0, 0.0)
    sat_frac = float((((m.detach().abs() > margin_sat_threshold) & valid).sum()
                      / valid.sum().clamp(min=1)).item())
    per_pair = -torch.nn.functional.logsigmoid(beta * m) * pair_weight
    loss = (per_pair * valid).sum() / valid.sum().clamp(min=1)
    n_valid = int(valid.sum().item())
    if not return_margins:
        return loss, n_valid, sat_frac
    return loss, n_valid, sat_frac, info
