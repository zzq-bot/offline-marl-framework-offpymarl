import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ITD3Critic(nn.Module):
    def __init__(self, scheme, args):
        super(ITD3Critic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        if getattr(self.args,"critic_rnn", False):
            self.critic_rnn = True
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.critic_rnn = False
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
    
    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        if self.critic_rnn:
            self.hidden_states = self.fc1.weight.new(batch_size, self.n_agents, self.args.hidden_dim).zero_()   
            self.hidden_states = self.hidden_states.reshape(-1, self.args.hidden_dim)

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        x = F.relu(self.fc1(inputs))
        if self.critic_rnn:
            bs = x.shape[0]
            x = x.reshape(-1, self.args.hidden_dim)
            h = self.rnn(x, self.hidden_states)
            q = self.fc3(x)
            self.hidden_states = h
            q = q.reshape(bs, -1, 1)
        else:
            x = F.relu(self.fc2(x))
            q = self.fc3(x)
        return q
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += self.n_actions
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape + self.n_actions 

