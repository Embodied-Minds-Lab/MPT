# Importing this module registers the three MPT Minecraft agents (Logger / Miner / Smith).
from .minecraft_chain_agent import LoggerDeskAgent, MinerDeskAgent, SmithDeskAgent, DESK_VERBS, first_verb

__all__ = ["LoggerDeskAgent", "MinerDeskAgent", "SmithDeskAgent", "DESK_VERBS", "first_verb"]
