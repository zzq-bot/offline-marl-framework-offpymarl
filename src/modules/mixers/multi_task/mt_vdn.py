import torch as th
import torch.nn as nn


class MTVDNMixer(nn.Module):
    def __init__(self):
        super(MTVDNMixer, self).__init__()

    def forward(self, agent_qs, batch, task=None):
        return th.sum(agent_qs, dim=2, keepdim=True)