REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .multi_task.mt_rnn_agent import MTRNNAgent
REGISTRY["mt_rnn"] = MTRNNAgent