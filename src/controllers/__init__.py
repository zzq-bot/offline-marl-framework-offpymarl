REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['random_mac'] = RandomMAC
REGISTRY['maddpg_mac'] = MADDPGMAC