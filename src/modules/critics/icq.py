import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ICQCritic(nn.Module):
    def __init__(self, scheme, args) -> None:
        super(ICQCritic, self).__init__()
        
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.critic_hidden_dim = args.critic_hidden_dim

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, self.critic_hidden_dim)
        self.fc2 = nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim)
        self.fc_v = nn.Linear(self.critic_hidden_dim, 1)
        self.fc3 = nn.Linear(self.critic_hidden_dim, self.n_actions)
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc3(x) # advantage
        q = a + v 
        return q

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        input_shape += scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape