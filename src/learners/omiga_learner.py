import copy
from modules.mixers.lmix import LMixer
from modules.critics.mlp import MLPCritic
import torch as th
from torch.distributions import Categorical
from torch.optim import RMSprop, Adam
from components.standarize_stream import RunningMeanStd

class OMIGALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.agent_params = list(mac.parameters())

        critic_input_shape = self._get_critic_input_shape(scheme)
        self.v_critic = MLPCritic(critic_input_shape, 1, args)
        self.q_critic = MLPCritic(critic_input_shape, self.args.n_actions, args)
        self.mixer = LMixer(args)
        
        self.v_params = list(self.v_critic.parameters())  
        self.q_params = list(self.q_critic.parameters()) + list(self.mixer.parameters())      
        
        
        match self.args.optim_type.lower():
            case "rmsprop":
                self.actor_optimiser = RMSprop(params=self.agent_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.q_optimiser = RMSprop(params=self.q_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.v_optimiser = RMSprop(params=self.v_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.actor_optimiser = Adam(params=self.agent_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
                self.q_optimiser = Adam(params=self.q_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
                self.v_optimiser = Adam(params=self.v_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.args.optim_type)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_v_critic = copy.deepcopy(self.v_critic)
        self.target_q_critic = copy.deepcopy(self.q_critic)
        self.target_mixer = copy.deepcopy(self.mixer)
        
        
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # termianted point 1 -> 0
        avail_actions = batch["avail_actions"]
        
        critic_inputs = self._build_critic_inputs(batch)
        
        
        cur_q_vals = self.q_critic(critic_inputs[:, :-1])
        cur_chosen_q_vals = th.gather(cur_q_vals, dim=3, index=actions)
        # print(cur_chosen_q_vals.shape, batch["state"].shape)
        cur_chosen_q_tot = self.mixer(cur_chosen_q_vals, batch["state"][:, :-1])
        
        next_v_vals = self.target_v_critic(critic_inputs[:, 1:]) # (b, T, n_agents, 1)
        next_w, next_b = self.target_mixer.w_and_b(batch["state"][:, 1:]) # (b, T, n_agents, 1). (b, T, 1)
        next_v_tot = (next_w * next_v_vals).sum(dim=-2) + next_b
        
        q_target = rewards + self.args.gamma * (1 - terminated) * next_v_tot.detach()
        q_error = (cur_chosen_q_tot - q_target) # (bs, T, 1)
        
        mask_q = mask.expand_as(q_error)
        
        q_loss = ((q_error * mask_q) ** 2).sum() / mask_q.sum()
        
        
        target_q_vals = self.target_q_critic(critic_inputs[:, :-1])
        targe_chosen_q_vals = th.gather(target_q_vals, dim=3, index=actions)
        target_w, _ = self.target_mixer.w_and_b(batch["state"][:, :-1])
        cur_v = self.v_critic(critic_inputs[:, :-1]) # (b, T, n_agents, 1)
        
        z = 1 / self.args.alpha_temp * (target_w.detach() * targe_chosen_q_vals.detach() - target_w.detach() * cur_v)
        z = th.clamp(z, min=-10.0, max=10.0)
        max_z = th.max(z)
        max_z = th.where(max_z < -1.0, th.tensor(-1.0).to(self.args.device), max_z)
        max_z = max_z.detach()
        
        
        v_error = th.exp(z - max_z) + th.exp(-max_z) * target_w.detach() * cur_v / self.args.alpha_temp
        mask_v = mask_q.unsqueeze(-1).expand_as(v_error)
       
        v_loss = (v_error * mask_v).sum() / mask_v.sum()
        
        exp_a = th.exp(z).detach().squeeze(-1)
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        mac_out[avail_actions==0] = 1e-10
       
        dist = Categorical(probs=mac_out[:, :-1])
        
        log_probs = dist.log_prob(actions.squeeze(-1)) # (bs, T, n_agents)
        mask_a = mask_q.expand_as(log_probs)
        
        actor_loss = -((exp_a * log_probs) * mask_a).sum() / mask_a.sum()
                
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.actor_optimiser.step()
        
        self.q_optimiser.zero_grad()
        q_loss.backward()
        th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()
        
        self.v_optimiser.zero_grad()
        v_loss.backward()
        th.nn.utils.clip_grad_norm_(self.v_params, self.args.grad_norm_clip)
        self.v_optimiser.step()
        
        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("v_loss", v_loss.item(), t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            #self.logger.log_stat("alpha_temp", self.args.alpha_temp, t_env)
            self.log_stats_t = t_env

    def _build_critic_inputs(self, batch):
        inputs  = []
        bs, max_t = batch.batch_size, batch.max_seq_length

        inputs.append(batch["obs"])
        assert batch.max_seq_length == batch["state"].shape[1]
        if self.args.obs_last_action:
            inputs.append(th.cat([th.zeros_like(batch["actions_onehot"][:, :1]), batch["actions_onehot"][:, :-1]], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1).to(self.args.device))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
    def _get_critic_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.target_v_critic.load_state_dict(self.v_critic.state_dict())
        
    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
       
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for target_param, param in zip(self.target_q_critic.parameters(), self.q_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for target_param, param in zip(self.target_v_critic.parameters(), self.v_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.q_critic.cuda()
        self.target_q_critic.cuda()
        self.v_critic.cuda()
        self.target_v_critic.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.q_critic.state_dict(), "{}/q_critic.th".format(path))  
        th.save(self.v_critic.state_dict(), "{}/v_critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.q_optimiser.state_dict(), "{}/q_opt.th".format(path))
        th.save(self.v_optimiser.state_dict(), "{}/v_opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
        self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
        self.q_critic.load_state_dict(th.load("{}/q_critic.th".format(path)))
        self.target_q_critic.load_state_dict(th.load("{}/q_critic.th".format(path)))
        self.v_critic.load_state_dict(th.load("{}/v_critic.th".format(path)))
        self.target_v_critic.load_state_dict(th.load("{}/v_critic.th".format(path)))
        self.actor_optimiser.load_state_dict(th.load("{}/actor_opt.th".format(path)))
        self.q_optimiser.load_state_dict(th.load("{}/q_opt.th".format(path)))
        self.v_optimiser.load_state_dict(th.load("{}/v_opt.th".format(path)))
                                            
    