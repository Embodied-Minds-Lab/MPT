from typing import List
import numpy as np
import re
from collections import defaultdict

from omegaconf import OmegaConf, DictConfig, ListConfig

def normalize_agent_id(agent_id: str) -> str:
    return agent_id.strip().replace(" ", "")

def normalize_model_id(model_id: str) -> str:
    return model_id.split("/")[-1]

def omega_equal_resolved(cfg1, cfg2) -> bool:
    """Check deep equality between two OmegaConf configs after resolving interpolations."""
    return OmegaConf.to_container(cfg1, resolve=True) == OmegaConf.to_container(cfg2, resolve=True)

def _set_specific_parameter(base_config, idx: int, total: int, agent_specific_parameters=None):
    """
    Update agent-specific parameters in the base configuration for a given agent index.

    Args:
        base_config: An OmegaConf.DictConfig or a regular dict.
        idx (int): The index of the current agent (0-based).
        total (int): Total number of agents.
        agent_specific_parameters: A dict mapping parameter paths to lists of values

    Returns:
        A new configuration (DictConfig) customized for the current agent.
    """

    def _extract_and_validate_lists(nested_dict, prefix="", total=total):
        """
        Recursively extract all list values from nested dict structure.
        Returns a flat dict mapping full paths to lists.
        """
        result = {}
        for key, value in nested_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (list, ListConfig)):
                # Found a list - validate and store
                if len(value) != total:
                    raise ValueError(
                        f"Agent-Specific Parameter '{current_path}' list length mismatch: "
                        f"expected {total} (number of agents), got {len(value)}. Value: {value}"
                    )
                result[current_path] = value
            elif isinstance(value, (dict, DictConfig)):
                # Recursively process nested dict
                result.update(_extract_and_validate_lists(value, current_path, total))
            else:
                raise ValueError(
                    f"Agent-Specific Parameter '{current_path}' must have a list or dict value, "
                    f"but got {type(value).__name__}. Value: {value}"
                )
        return result
    
    # Deep-copy the base config to avoid mutating the original
    cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    if agent_specific_parameters is None:
        return cfg

    if isinstance(agent_specific_parameters, (dict, DictConfig)):
        # Convert nested dict structure to flat path->list mapping
        flat_params = _extract_and_validate_lists(agent_specific_parameters, total=total)
        
        # Update config with selected values for this agent
        for param_path, values in flat_params.items():
            specific_value = values[idx]
            OmegaConf.update(cfg, param_path, specific_value, merge=False)
        return cfg

    raise TypeError(
        f"agent_specific_parameters must be a dict or None, "
        f"but got {type(agent_specific_parameters).__name__}"
    )

def build_wg_ids(config):
    agent_ids = config.agent.agent_ids
    model_ids = config.agent.model_ids
    model_sharing = config.agent.model_sharing

    if len(agent_ids) != len(model_ids):
        raise ValueError("agent_ids and model_ids must have the same length.")


    base_config = config.actor_rollout_ref
    agent_specific_parameters = config.agent.agent_specific_parameters
    
    total_agent_num = len(agent_ids)
    if not model_sharing:
        wg_to_agents = {}
        agent_port = 0
        for idx in range(total_agent_num):
            agent_id = agent_ids[idx]
            model_id = model_ids[idx]
            per_config = _set_specific_parameter(base_config, idx, total_agent_num, agent_specific_parameters)
            per_config.rollout["agent_port"] = agent_port
            norm_agent = normalize_model_id(agent_id)
            norm_model = normalize_model_id(model_id)
            wg_id = f"{norm_model}_{norm_agent}"
            wg_to_agents[wg_id] = [{"agent_id": agent_id, "model_id": model_id, "config_actor_rollout_ref": per_config}]
            agent_port += 1
    else:
        model_to_agents = defaultdict(list)
        for idx in range(total_agent_num):
            agent_id = agent_ids[idx]
            model_id = model_ids[idx]
            per_config = _set_specific_parameter(base_config, idx, total_agent_num, agent_specific_parameters)
            model_to_agents[model_id].append({"agent_id": agent_id, "config_actor_rollout_ref": per_config})

        wg_to_agents = {}
        agent_port = 0
        for model_id, agents_configs in model_to_agents.items():
            for ac in agents_configs:
                ac["config_actor_rollout_ref"].rollout["agent_port"] = agent_port
            ref_cfg = agents_configs[0]["config_actor_rollout_ref"]
            for ac in agents_configs[1:]:
                assert omega_equal_resolved(ref_cfg, ac["config_actor_rollout_ref"]), (
                    f"Agents sharing model '{model_id}' must have equal configs."
                )

            norm_agents = [normalize_agent_id(ac["agent_id"]) for ac in agents_configs]
            norm_model = normalize_model_id(model_id)
            wg_id = "_".join([norm_model] + norm_agents)
            wg_to_agents[wg_id] = [
                {
                    "agent_id": ac["agent_id"],
                    "model_id": model_id,
                    "config_actor_rollout_ref": ac["config_actor_rollout_ref"]
                }
                for ac in agents_configs
            ]
            agent_port += 1

    return wg_to_agents

def general_projection(text_repsonses: List[str], start_tag: str, end_tag: str, check_think_tag: bool = False, return_tag: bool = False, return_whole_response: bool = False) -> List[str]:
    """
    An function to process the text_repsonses
    text_repsonses: the list of text_repsonses to be processeed, it is a list of strings.
    start_tag: the start tag to be used for projection, e.g., "<action>"
    end_tag: the end tag to be used for projection, e.g., "</action>
    check_think_tag: whether to check the <think>...</think> tag, default is False.
    return_tag: whether to return the tag, default is False.
    return_whole_response: whether to return the whole response, default is False.
    """
    valids = [0] * len(text_repsonses)

    for i in range(len(text_repsonses)):
        original_str = text_repsonses[i]  # keep the original string
        start_idx = text_repsonses[i].find(start_tag)
        end_idx = text_repsonses[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                valids[i] = 0
                if not return_whole_response:
                    text_repsonses[i] = ""
                continue
            
            extracted_action = text_repsonses[i][start_idx + len(start_tag):end_idx].strip()
            if return_whole_response:
                text_repsonses[i] = original_str[:start_idx + len(start_tag)] + extracted_action + original_str[end_idx:]
            else:
                if return_tag:
                    text_repsonses[i] = start_tag + extracted_action + end_tag
                else:
                    text_repsonses[i] = extracted_action

            valids[i] = 1

        except:
            valids[i] = 0
            if not return_whole_response:
                text_repsonses[i] = ""

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

        if check_think_tag:
            # check <think>...</think>
            # Must have exactly one start tag and one end tag, and end tag must be after start tag
            think_start_count = original_str.count("<think>")
            think_end_count = original_str.count("</think>")
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if (think_start_count != 1 or think_end_count != 1 or 
                think_start_idx == -1 or think_end_idx == -1 or 
                think_end_idx <= think_start_idx):
                valids[i] = 0

    valids = np.array(valids, dtype=bool)
    return text_repsonses, valids


_CHAIN_OPEN_BID_RE = re.compile(r"<bid>\s*([-+]?[0-9]*\.?[0-9]+)", re.IGNORECASE)
_PARTIAL_CLOSE_RE = re.compile(r"<\s*/?\s*b?i?d?\s*$", re.IGNORECASE)


def parse_open_bid(text_responses: List[str], bid_max: float = 1.0):
    """Parse the unified-IO phase-A probe: generation stops AT '</bid>' (stop string excluded), so
    the text looks like '<bid>0.4'. Returns (bids[float32], has_bid[bool], clean_prefix[list[str]]).
    clean_prefix = the probe text with any trailing PARTIAL '</bid' fragment stripped (produced when
    the probe hits max_new_tokens mid-tag) — ready to have a canonical '</bid>' appended for the
    winner's continuation."""
    n = len(text_responses)
    bids = np.zeros(n, dtype=np.float32)
    has_bid = np.zeros(n, dtype=bool)
    clean = []
    bmax = float(bid_max)
    for i, s in enumerate(text_responses):
        s = s if isinstance(s, str) else ""
        m = _CHAIN_OPEN_BID_RE.search(s)
        if m is not None:
            try:
                bids[i] = float(np.clip(float(m.group(1)), 0.0, bmax))
                has_bid[i] = True
            except Exception:
                bids[i] = 0.0
        # defensive: if the engine ever returns the stop string (or anything after it), cut there
        if "</bid>" in s:
            s = s.split("</bid>", 1)[0]
        clean.append(_PARTIAL_CLOSE_RE.sub("", s.rstrip()))
    return bids, has_bid, clean
