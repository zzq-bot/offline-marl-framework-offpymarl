import copy
import torch as th
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import RMSprop, Adam

from modules.mixers.qmix import QMixer
from modules.critics.icq import ICQCritic
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets


class ICQLearner:
    def __init__(self, mac, scheme, logger, args) -> None:
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.icq_alpha = args.icq_alpha
        self.icq_beta = args.icq_beta

        self.mac = mac
        self.agent_params = list(mac.parameters())

        self.critic = ICQCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = QMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.mixer_params = list(self.mixer.parameters())

        self.c_params = self.critic_params + self.mixer_params

        match self.args.optim_type.lower():
            case "rmsprop":
                self.agent_optimiser = RMSprop(params=self.agent_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.critic_optimiser = RMSprop(params=self.critic_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.mixer_optimiser = RMSprop(params=self.mixer_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
                self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
                self.mixer_optimiser = Adam(params=self.mixer_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.args.optim_type)


        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # train critic first
        critic_log = self.train_critic(batch)

        bs = batch.batch_size
        max_t = batch.max_seq_length

        states = batch["state"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mask = mask.repeat(1, 1, self.n_agents).view(bs, -1, self.n_agents)

        critic_inputs = self._build_critic_inputs(batch)
        q_vals = self.critic(critic_inputs).detach()[:, :-1]# (bs, seq_len-1, n_agents, n_actions)

        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_t-1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1) # (bs, seq_len-1, n_agents, n_actions)

        mac_out[avail_actions == 0] = 0 # already softmax
        pi = mac_out / mac_out.sum(dim=-1, keepdim=True) # normalize again over actions
        pi[avail_actions == 0] = 0
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        # (bs , (T-1) , n_agents,)
        
        pi_taken[mask == 0] = 1.0
        assert not th.isnan(pi_taken).any()
        log_pi_taken = th.log(pi_taken)
        
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)

        baseline = th.sum(mac_out * q_vals, dim=-1).detach() # (bs, T-1, n_agents)

        coe = self.mixer.k(states)

        advantages = (q_taken - baseline)
        advantages = F.softmax(advantages / self.icq_alpha, dim=0)

        actor_loss = - (coe * (bs * advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        assert len(advantages) == bs

        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.training_steps += 1
        
        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("actor_grad_norm", actor_grad_norm.item(), t_env)
            for k, v in critic_log.items():
                self.logger.log_stat(k, v, t_env)
            self.log_stats_t = t_env


    def train_critic(self, batch: EpisodeBatch):
        critic_log = {  "critic_loss":[],
                        "critic_grad_norm":[],  
                        "td_error_abs":[], 
                        "target_mean":[], 
                        "q_taken_mean":[]}
        bs = batch.batch_size
        max_t = batch.max_seq_length

        states = batch["state"]
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])


        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
        critic_inputs = self._build_critic_inputs(batch)
        target_q_vals = self.target_critic(critic_inputs).detach() # (bs, seq_len, n_agents, n_action)

        with th.no_grad():
            # -----------------------------Q_lambda-IS-----------------------
            target_q_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(-1)
            target_q_vals_IS = self.target_mixer.icq_forward(target_q_taken, states) # (bs, seq_len, 1)
            advantage_Q = F.softmax(target_q_vals_IS / self.icq_beta, dim=0) # softmax over trajectory
            targets_taken = self.target_mixer.icq_forward(target_q_taken, states)
            targets_taken = bs * advantage_Q * targets_taken # do not why there is a "bs*" operator
            assert bs == len(advantage_Q)
            target_q = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
            target_q = target_q.detach()
            # (bs, seq_len-1, 1)
        for t in range(max_t - 1):
            mask_t = mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals = self.critic(critic_inputs[:, t:t+1])
            q_vals = th.gather(q_vals, 3, index=actions[:, t:t+1]).squeeze(3)
            q_vals = self.mixer.icq_forward(q_vals, states[:, t:t+1])

            target_q_t = target_q[:, t:t+1].detach()
            q_err = (q_vals - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum()

            self.critic_optimiser.zero_grad()
            self.mixer_optimiser.zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.mixer_optimiser.step()
    
            critic_log["critic_loss"].append(critic_loss.item())
            critic_log["critic_grad_norm"].append(critic_grad_norm.item())
            mask_elems = mask_t.sum().item()
            critic_log["td_error_abs"].append(((q_err.abs().sum().item() / mask_elems)))
            critic_log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            critic_log["q_taken_mean"].append((q_vals* mask_t).sum().item() / mask_elems)
        for k in critic_log.keys():
            critic_log[k] = np.mean(critic_log[k])
        return critic_log
        
        # not stable
        q_vals = self.critic(critic_inputs[:, :max_t-1])
        chosen_action_q_vals = th.gather(q_vals, 3, index=actions[:, :max_t-1])
        chosen_action_q_vals = self.mixer.icq_forward(chosen_action_q_vals, states[:, :max_t-1]) 
        # (bs, seq_len, 1)
        filter_t = th.where(mask.sum(dim=0)>=0.5)[0]
        td_error = (chosen_action_q_vals - target_q) * mask
        td_error = td_error[:, filter_t]
        critic_loss = (td_error ** 2).sum() / mask[:, filter_t].sum()

        self.critic_optimiser.zero_grad()
        self.mixer_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.mixer_optimiser.step()

        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()
        mask_elems = mask[:, filter_t].sum().item()
        critic_log["td_error_abs"] = ((td_error.abs().sum().item() / mask_elems))
        critic_log["target_mean"] = (target_q[:, filter_t] * mask[:, filter_t]).sum().item() / mask_elems
        critic_log["q_taken_mean"] = (chosen_action_q_vals[:, filter_t] * mask[:, filter_t]).sum().item() / mask_elems
        return critic_log

    def _build_critic_inputs(self, batch):
        inputs  = []
        bs, max_t = batch.batch_size, batch.max_seq_length
        inputs.append(batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1)) # (bs, seq_len, n_agents, state_dim)
        inputs.append(batch["obs"])
        assert batch.max_seq_length == batch["state"].shape[1]
        inputs.append(th.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1).to(self.args.device))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    
    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        assert tau <= 1
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.mixer.cuda()
        self.target_critic.cuda()
        self.target_mixer.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic = copy.deepcopy(self.critic)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_optimiser.load_state_dict(
            th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))
    