from .coma import COMACritic
from .maddpg import MADDPGCritic
from .icq import ICQCritic
from .itd3 import ITD3Critic
from .double import DoubleCritic

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["icq_critic"] = ICQCritic
REGISTRY["itd3_critic"] = ITD3Critic
REGISTRY["double_critic"] = DoubleCritic

from .multi_task.mt_maddpg import MTMADDPGCritic
REGISTRY["mt_maddpg_critic"] = MTMADDPGCritic