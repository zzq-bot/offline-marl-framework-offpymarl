import torch as th
import torch.nn as nn
import numpy as np


class LMixer(nn.Module):
    def __init__(self, args):
        super(LMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        
        self.f_v = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w = nn.Linear(self.embed_dim, self.n_agents)
        self.hyper_b = nn.Linear(self.embed_dim, 1)
    
    def forward(self, agent_qs, states):
        # agent_qs.shape (bs, T, n_agents, 1)
        bs, T = states.shape[:2]
        states = states.reshape(-1, self.state_dim) # (bs, T, state_dim)
        x = self.f_v(states) # (bs, T, embed_dim)
        
        w = th.abs(self.hyper_w(x).reshape(bs, T, self.n_agents, 1))
        b = self.hyper_b(x).reshape(bs, T, 1)

        q_tot = (agent_qs * w).sum(dim=2) + b # (bs, T, 1)
        return q_tot        

    def w_and_b(self, states):
        bs, T = states.shape[:2]
        x = self.f_v(states)
        w = th.abs(self.hyper_w(x).reshape(bs, T, self.n_agents, 1))
        b = self.hyper_b(x).reshape(bs, T, 1)
        return w, b
    