import torch.nn as nn
import torch.nn.functional as F

class MLPCritic(nn.Module):
    def __init__(self, input_shape, output_shape, args) -> None:
        super(MLPCritic, self).__init__()
        
        self.args = args
        self.critic_hidden_dim = args.critic_hidden_dim

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, self.critic_hidden_dim)
        self.fc2 = nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim)
        self.fc3 = nn.Linear(self.critic_hidden_dim, output_shape)

    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        val = self.fc3(x)
        return val