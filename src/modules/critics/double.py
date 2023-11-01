import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DoubleCritic(nn.Module):
    def __init__(self, scheme, args):
        super(DoubleCritic, self).__init__()
        
        self.args = args
        self.n_actions = args.n_actions
        assert scheme["actions_onehot"]["vshape"][0] == self.n_actions
        self.n_agents = args.n_agents

        self.input_dim = self._get_input_shape(scheme)
        
        self.output_type = "q"
        
        self.fc1 = nn.Linear(self.input_dim, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc3 = nn.Linear(args.critic_hidden_dim, self.n_actions)
        # TODO add optional params, whether dueling
        self.fc3_v = nn.Linear(args.critic_hidden_dim, 1)

        self.fc4 = nn.Linear(self.input_dim, args.critic_hidden_dim)
        self.fc5 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.fc6 = nn.Linear(args.critic_hidden_dim, self.n_actions)
        self.fc6_v = nn.Linear(args.critic_hidden_dim, 1)
    
    def forward(self, X):
        assert len(X.shape) == 4
        bs, t, n_agents, _ = X.shape
        X = X.reshape(-1, self.input_dim)

        h1 = F.relu(self.fc1(X))
        h2 = F.relu(self.fc2(h1))
        a1 = self.fc3(h2) 
        v1 = self.fc3_v(h2)
        q1 = (a1+v1).reshape(bs, t, n_agents, -1)

        h1_2 = F.relu(self.fc4(X))
        h2_2 = F.relu(self.fc5(h1_2))
        a2 = self.fc3(h2_2)
        v2 = self.fc3_v(h2_2)
        q2 = (a2+v2).reshape(bs, t, n_agents, -1)
        return q1, q2
    
    def build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state, obs, action
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:, ts])
        inputs.append(th.eye(self.n_agents, device=self.args.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        # TODO add optional params about action
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        input_shape += self.n_agents
        # TODO add optional params about action
        return input_shape