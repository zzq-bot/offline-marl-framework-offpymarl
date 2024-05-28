import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        


    def init_hidden(self):
        # make hidden states on same device as model
        return None

    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
