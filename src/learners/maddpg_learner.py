import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd


class MADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        match self.args.optim_type.lower():
            case "rmsprop":
                self.agent_optimiser = RMSprop(params=self.agent_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.critic_optimiser = RMSprop(params=self.critic_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
                self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
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
    
    def train(self, batch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        actions = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        mask = 1 - terminated
        batch_size = batch.batch_size

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
        # Train the critic
        critic_inputs = self._build_critic_inputs(batch)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        q_taken = self.critic(critic_inputs[:, :-1], actions[:, :-1].detach()) # (batch_size, T-1, n_agents, 1)
        q_taken = q_taken.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)

        # Use the target actor and target critic network to compute the target q
        self.target_mac.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t) # (bs, n_agents, n_ac)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)[:, 1:]  # Concat over time, (bs, T-1, n_agents, n_ac)

        target_actions = target_actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        target_vals = self.target_critic(critic_inputs[:, 1:], target_actions.detach())
        target_vals = target_vals.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals.reshape(-1, 1).detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        
        td_error = (q_taken.view(-1, 1) - targets.detach())
        masked_td_error  = td_error * mask.reshape(-1, 1)
        #critic_loss = (masked_td_error ** 2).sum()/mask.sum()
        critic_loss = (masked_td_error ** 2).mean()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # Train the actor
        self.mac.init_hidden(batch_size)
        pis = []
        actions = []
        for t in range(batch.max_seq_length-1):
            pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
            pis.append(pi)
            actions.append(gumbel_softmax(pi, hard=True)) # (batch_size, 1, n_agents, n_acs)
        
        actions = th.cat(actions, dim=1) # (batch_size, T-1, n_agents, n_acs)
        actions = actions.view(batch_size, -1, 1, self.n_agents*self.n_actions) # (bs, T-1, 1, n_agents*n_actions)
        actions = actions.expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, n_agents*n_actions)

        new_actions = []
        for i in range(self.n_agents):
            temp_action = th.split(actions[:, :, i, :], self.n_actions, dim=2)
            # len(temp_action)=self.n_agents, temp[action][0].shape==(bs, T-1, n_actions)
            actions_i = []
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j]) # keep the gradient of itself
                else:
                    actions_i.append(temp_action[j].detach()) # detach others when calculate q^i(s, a_i, a_{-i}.detach)
            actions_i = th.cat(actions_i, dim=-1) # (bs, T-1, n_agents*n_actions)
            new_actions.append(actions_i.unsqueeze(2))
        new_actions = th.cat(new_actions, dim=2) # (bs, T-1, n_agents, n_agents*n_actions)

        pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
        pis[pis==-1e10] = 0
        masked_pis = pis * mask.expand_as(pis)
        masked_pis = masked_pis.reshape(-1, 1) # (bs * (T-1) * n_agents, n_actions)
        q = self.critic(critic_inputs[:, :-1], new_actions)
        q = q.reshape(-1, 1) # (bs * (T-1) * n_agents, 1)
        mask = mask.reshape(-1, 1)

        # Compute the actor loss
        # actor_loss = - (q * mask).sum() / mask.sum() + ...
        # original: pis ** 2 -> masked_pis ** 2
        """print(q.shape)
        print(mask.shape)
        print(pi.shape)"""
        actor_loss = - (q * mask).mean() + self.args.reg * ( masked_pis ** 2).mean()

        # Optimise agents
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
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q_taken).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("actor_grad_norm", actor_grad_norm.item(), t_env)
            self.log_stats_t = t_env
                
    def _build_critic_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1))

        if self.args.critic_individual_obs:
            inputs.append(batch["obs"][:, ts])

        if self.args.critic_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                #assert all(batch["actions_onehot"][:, slice(t - 1, t)]==batch["actions_onehot"][:, t-1:t])
                inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                        batch["actions_onehot"][:, :-1]], dim=1)
                inputs.append(last_actions)

        if self.args.critic_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, self.n_agents, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs



    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic = copy.deepcopy(self.critic)
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))