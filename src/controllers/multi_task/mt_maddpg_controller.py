from modules.agents import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY
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


class MTMADDPGMAC:
    def __init__(self, all_tasks, task2scheme, task2args, main_args):
        self.all_tasks = all_tasks
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.task2n_agents = {task: self.task2args[task].n_agents for task in all_tasks}
        self.task2n_actions = {task: self.task2args[task].n_actions for task in all_tasks}
        self.main_args = main_args

        self.agent_output_type = main_args.agent_output_type
        self.action_selector = None # Will not be used

        if self.main_args.env != "sc2":
            raise NotImplementedError("Support for envs except for smac will be updated later")
        env2decomposer = {
            "sc2": "sc2_decomposer",
        }
        self.task2decomposer = {}
        self.surrogate_decomposer = None
        aligned_unit_type_bits, aligned_shield_bits_ally, aligned_shield_bits_enemy = 0, 0, 0
        map_type_set = set()
        for task in all_tasks:
            task_args = self.task2args[task]
            if not task_args.env == "sc2":
                raise NotImplementedError(f"Unsupported env decomposer {task_args.env}")
            task_decomposer = decomposer_REGISTRY[env2decomposer[task_args.env]](task_args)
            
            aligned_shield_bits_ally = max(aligned_shield_bits_ally, task_decomposer.shield_bits_ally)
            aligned_shield_bits_enemy = max(aligned_shield_bits_enemy, task_decomposer.shield_bits_enemy)
            #unit_types = get_unit_type_from_map_type(task_decomposer.map_type)
            for unit_type in task_decomposer.unit_types:
                map_type_set.add(unit_type)

            #task_decomposer._print_info()
            self.task2decomposer[task] = task_decomposer
            # set obs_shape, state_dim
            #task_args.obs_shape = task_decomposer.obs_dim
            #task_args.state_shape = task_decomposer.state_dim
        aligned_unit_type_bits = 0 if len(map_type_set) == 1 else len(map_type_set)
        for task in all_tasks:
            self.task2decomposer[task].align_feats_dim(aligned_unit_type_bits, aligned_shield_bits_ally, aligned_shield_bits_enemy, map_type_set)
            if not self.surrogate_decomposer:
                self.surrogate_decomposer = self.task2decomposer[task]
            task_args.obs_shape = self.task2decomposer[task].aligned_obs_dim
            task_args.state_shape = self.task2decomposer[task].aligned_state_dim
        
        # build agents
        
        self.task2input_shape_info = self._get_input_shape()
        self._build_agents(self.task2input_shape_info)
        
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, task=None, test_mode=False):
        assert task is not None
        agent_outputs = self.forward(ep_batch, t_ep, task)
        if test_mode:
            chosen_actions = onehot_from_logits(agent_outputs).argmax(dim=-1)
        else:
            chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions
    
    def target_actions(self, ep_batch, t_ep, task, agent_id=None):
        agent_outputs = self.forward(ep_batch, t_ep, task, agent_id)
        return onehot_from_logits(agent_outputs)
    
    def forward(self, ep_batch, t, task, agent_id=None):
        agent_inputs = self._build_inputs(ep_batch, t, task, agent_id)
        avail_actions = ep_batch["avail_actions"][:, t]
        # bs = agent_inputs.shape[0]//self.task2n_agents[task]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, task)
        if agent_id is None:
            agent_outs = agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs
    
    
    def init_hidden(self, batch_size, task):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.task2n_agents[task], -1)

    def init_hidden_one_agent(self, batch_size):
        self.hidden_states = self.agent.init_hidden().expand(batch_size, -1) # (bv)

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

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.main_args.agent](task2input_shape_info,
                                                          self.task2decomposer, self.task2n_agents,
                                                          self.surrogate_decomposer, self.main_args)
    
    def _build_inputs(self, batch, t, task, agent_id=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if agent_id is not None:
            raise NotImplementedError("TBD")
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        if task_args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if task_args.obs_agent_id:
            inputs.append(th.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        inputs = th.cat([x.reshape(bs*n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.all_tasks:
            task_scheme = self.task2scheme[task]
            obs_shape = task_scheme["obs"]["vshape"]
            input_shape = obs_shape
            last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            #joint_action_shape = task_scheme["actions_onehot"]["vshape"][0] * self.task2n_agents[task]
            agent_id_shape = self.task2n_agents[task]
            if self.task2args[task].obs_last_action:
                input_shape += last_action_shape
            if self.task2args[task].obs_agent_id:
                input_shape += agent_id_shape

            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "obs_shape": obs_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
                #"joint_action_shape": joint_action_shape,
            }
        return task2input_shape_info