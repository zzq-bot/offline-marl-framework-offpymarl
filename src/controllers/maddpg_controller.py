from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs #.shape == logits.shape

def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, return_dist=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y_keep_gradient = (y_hard - y).detach() + y # keep gradient
    if return_dist:
        return y_keep_gradient, y
    return y_keep_gradient


# This multi-agent controller shares parameters between agents
class MADDPGMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = None

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        if test_mode:
            chosen_actions = onehot_from_logits(agent_outputs).argmax(dim=-1)
        else:
            chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep, agent_id=None):
        agent_outputs = self.forward(ep_batch, t_ep, agent_id)
        return onehot_from_logits(agent_outputs) # (bs, n_agents, n_ac) no need to softmax/gumble sampling due to detach()

    def forward(self, ep_batch, t, agent_id=None):
        agent_inputs = self._build_inputs(ep_batch, t, agent_id)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if agent_id is None:
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs

    def get_repeat_actions(self, ep_batch, t, num_repeats=None):
        if num_repeats is not None:
            agent_inputs = self._build_inputs(ep_batch, t).reshape(ep_batch.batch_size, self.n_agents, -1) # (bs, n_agents, x)
            input_dim = agent_inputs.shape[-1]
            agent_inputs = agent_inputs.unsqueeze(2).repeat(1, 1, num_repeats, 1).reshape(-1, input_dim)
            avail_actions = ep_batch["avail_actions"][:, t].unsqueeze(2).repeat(1, 1, num_repeats, 1)
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, num_repeats, -1)
        else:
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions==0] = -1e10
        chosen_actions, dist = gumbel_softmax(agent_outs, hard=True, return_dist=True)
        dist = Categorical(dist) # (bs, num_repeats, n_agents, n_actions)
        chosen_actions_index = chosen_actions.argmax(dim=-1)
        return chosen_actions, dist.log_prob(chosen_actions_index)
    

    def init_hidden(self, batch_size, num_repeats=None):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        if num_repeats is not None:
            self.hidden_states = self.hidden_states.unsqueeze(2).repeat(1, 1, num_repeats, 1)
    
    def init_hidden_one_agent(self, batch_size): 
        self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)  # bv

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t, agent_id=None):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])

        if agent_id is None:
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        else:
            if self.args.obs_agent_id:
                auxiliary_input = th.zeros(bs, self.n_agents, device=batch.device)
                auxiliary_input[:, agent_id] = 1
                inputs.append(auxiliary_input)
            inputs = th.cat(inputs, dim=1)
        return inputs
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape