from verl import DataProto
import torch
import numpy as np

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizers, num_examine, normalize_by_length=False, profit_as_reward=False) -> None:
        self.tokenizers = tokenizers
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        # MPT: use each agent turn's settled revenue-share PROFIT minus rent (econ_credit) as its
        # outcome reward instead of the shared episode reward. Train-time only; validation keeps R.
        self.profit_as_reward = profit_as_reward

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            agent_id = data_item.non_tensor_batch['agent_id']
            wg_id = data_item.non_tensor_batch['wg_id']
            # decode
            prompt_str = self.tokenizers[wg_id].decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizers[wg_id].decode(valid_response_ids, skip_special_tokens=False)

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # blind rows on a VLM carry an EMPTY multi_modal_inputs placeholder — guard on content
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs:
                pixel_values = multi_modal_inputs.get('pixel_values')
                image_grid_thw = multi_modal_inputs.get('image_grid_thw')


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.profit_as_reward and 'econ_credit' in data_item.non_tensor_batch:
                # MPT: the agent's settled profit (minus rent) is its outcome reward.
                score = float(data_item.non_tensor_batch['econ_credit'])
            elif self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            tag= f"{data_source}{agent_id}"
            if tag not in already_print:
                already_print[tag] = 0

            if already_print[tag] < self.num_examine and np.random.random() < 0.1:
                already_print[tag] += 1
                print(f"[{data_source}][{agent_id}][prompt]", prompt_str)
                print(f"[{data_source}][{agent_id}][response]", response_str)
                print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
