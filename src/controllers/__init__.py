REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['random_mac'] = RandomMAC
REGISTRY['maddpg_mac'] = MADDPGMAC

from .multi_task.mt_basic_controller import MTBasicMAC
from .multi_task.mt_maddpg_controller import MTMADDPGMAC

REGISTRY["mt_basic_mac"] = MTBasicMAC
REGISTRY['mt_maddpg_mac'] = MTMADDPGMAC