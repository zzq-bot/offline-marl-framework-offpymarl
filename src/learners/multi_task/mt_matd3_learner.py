import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from controllers.maddpg_controller import gumbel_softmax
from modules.critics.multi_task.mt_maddpg import MTMADDPGCritic

class MTMATD3Learner:
    def __init__(self, mac, logger, main_args) -> None:
        self.main_args = main_args
        self.actor_freq = main_args.actor_freq
        self.logger = logger

        
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.task2n_actions= mac.task2n_actions
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer
        self.task2input_shape_info = mac.task2input_shape_info
        
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())

        self.critic1 = MTMADDPGCritic(self.task2input_shape_info, self.task2decomposer, self.task2n_agents, self.surrogate_decomposer, main_args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2 = MTMADDPGCritic(self.task2input_shape_info, self.task2decomposer, self.task2n_agents, self.surrogate_decomposer, main_args)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())

        match self.main_args.optim_type.lower():
            case "rmsprop":
                self.agent_optimiser = RMSprop(params=self.agent_params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
                self.critic_optimiser = RMSprop(params=self.critic_params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
            case "adam":
                self.agent_optimiser = Adam(params=self.agent_params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
                self.critic_optimiser = Adam(params=self.critic_params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        
        self.task2train_info = {}
        for task, task_args in self.task2args.items():
            self.task2train_info[task] = {
                "log_stats_t": -task_args.learner_log_interval - 1,
                "training_steps": 0
            }

        self.total_training_steps = 0
        self.last_target_update_episode = 0
        self.task_log_actor = {}
        for task in self.task2args:
            self.task_log_actor[task] = {
                "actor_loss": [],
                "actor_grad_norm": [],
            }
            if "bc" in self.main_args.name:
                self.task_log_actor[task]["bc_loss"] = []
                self.task_log_actor[task]["td3_loss"] = []

        device = "cuda" if main_args.use_cuda else "cpu"
        if self.main_args.standardise_returns:
            self.task2ret_ms = {}
            for task in self.task2args.keys():
                self.task2ret_ms[task] = RunningMeanStd(shape=(self.task2n_agents[task], ), device=device)
        if self.main_args.standardise_rewards:
            self.task2rew_ms = {}
            for task in self.task2args.keys():
                self.task2rew_ms[task] = RunningMeanStd(shape=(1, ), device=device)
            
    
    def train(self, batch, t_env: int, episode_num: int, task: str):
        task_critic_log = self.train_critic(batch, task)
        
        if (self.total_training_steps + 1) % self.actor_freq == 0:
            batch_size = batch.batch_size
        
            critic_states_inputs, critic_individual_inputs = self._build_critic_inputs(batch, task)
            actions_4bc = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            terminated = terminated.unsqueeze(2).expand(-1, -1, self.task2n_agents[task], -1) # (bs, T-1, n_agents, 1)
            mask = 1 - terminated
            # Train the actor
            self.mac.init_hidden(batch_size, task)
            pis = []
            actions = []
            for t in range(batch.max_seq_length-1):
                pi = self.mac.forward(batch, t=t, task=task).view(batch_size, 1, self.task2n_agents[task], -1) # (batch_size, 1, n_agents, n_acs)
                pis.append(pi) # avail_actions have been masked already
                actions.append(gumbel_softmax(pi, hard=True)) # (batch_size, 1, n_agents, n_acs)
            
            actions = th.cat(actions, dim=1) # (batch_size, T-1, n_agents, n_acs)
            actions = actions.view(batch_size, -1, 1, self.task2n_agents[task]*self.task2n_actions[task]) # (bs, T-1, n_agents*n_actions)
            actions = actions.expand(-1, -1, self.task2n_agents[task], -1) # (bs, T-1, n_agents, n_agents*n_actions)

            new_actions = []
            for i in range(self.task2n_agents[task]):
                temp_action = th.split(actions[:, :, i, :], self.task2n_actions[task], dim=2)
                # len(temp_action)=self.task2n_agents[task], temp[action][0].shape==(bs, T-1, n_actions)
                actions_i = []
                for j in range(self.task2n_agents[task]):
                    if i == j:
                        actions_i.append(temp_action[j]) # keep the gradient of itself
                    else:
                        actions_i.append(temp_action[j].detach()) # detach others when calculate q^i(s, a_i, a_{-i}.detach)
                actions_i = th.cat(actions_i, dim=-1) # (bs, T-1, n_agents*n_actions)
                new_actions.append(actions_i.unsqueeze(2))
            new_actions = th.cat(new_actions, dim=2) # (bs, T-1, n_agents, n_agents*n_actions)

            q = self.critic1(critic_states_inputs[:, :-1], new_actions, critic_individual_inputs[:, :-1], task, actor_update=True)
            q = q.reshape(-1, 1) # (bs * (T-1) * n_agents, 1)

            if "bc" in self.main_args.name: # matd3+bc
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis_mask = mask.expand_as(pis)
                pis = pis.reshape(-1, self.task2n_actions[task])
                #pis = pis * pis_mask.reshape(-1, self.task2n_actions[task])
                bc_loss = F.cross_entropy(pis, actions_4bc.reshape(-1), reduction="sum")
                bc_loss = bc_loss/(pis_mask.sum())
                
                mask = mask.reshape(-1, 1)
                #lmbda = self.main_args.td3_alpha / ((q * mask).sum()/mask.sum()).abs().mean().detach()
                lmbda = self.main_args.td3_alpha / ((q * mask).abs().sum().detach() / mask.sum()) 
                td3_loss = - lmbda * (q * mask).mean() 
                
                actor_loss = td3_loss + bc_loss
            else: 
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis[pis==-1e10] = 0
                masked_pis = pis * mask.expand_as(pis)
                masked_pis = masked_pis.reshape(-1, 1) # (bs * (T-1) * n_agents, n_actions)

                mask = mask.reshape(-1, 1)
                actor_loss = - (q * mask).mean() + self.main_args.reg * ( masked_pis ** 2).mean()
                #actor_loss = bc_loss
            # Optimise agents
            self.agent_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.main_args.grad_norm_clip)
            self.agent_optimiser.step()

            if self.main_args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.main_args.target_update_interval_or_tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
            elif self.main_args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.main_args.target_update_interval_or_tau)
            # save record
            self.task_log_actor[task]["actor_loss"].append(actor_loss.item())
            self.task_log_actor[task]["actor_grad_norm"].append(actor_grad_norm.item())
            if "bc" in self.main_args.name:
                self.task_log_actor[task]["bc_loss"].append(bc_loss.item())
                self.task_log_actor[task]["td3_loss"].append(td3_loss.item())

        self.total_training_steps += 1
        self.task2train_info[task]["training_steps"] += 1
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            for k, v in task_critic_log.items():
                self.logger.log_stat(f"{task}/{k}", v, t_env)
            if len(self.task_log_actor[task]["actor_loss"]) > 0:
                ts = len(self.task_log_actor[task]["actor_loss"])
                for k, v in self.task_log_actor[task].items():
                    self.logger.log_stat(f"{task}/{k}", sum(v)/ts, t_env)
                    self.task_log_actor[task][k].clear()

            self.log_stats_t = t_env

    def train_critic(self, batch, task:str):
        critic_log = {}
        batch_size = batch.batch_size

    
        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.task2n_agents[task], -1) # (bs, T-1, n_agents, 1)
        
        actions = batch["actions_onehot"] # (bs, T, n_agents, n_actions)
        terminated = batch["terminated"][:, :-1].float()
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.task2n_agents[task], -1) # (bs, T-1, n_agents, 1)
        mask = 1 - terminated

        if self.main_args.standardise_rewards:
            self.task2rew_ms[task].update(rewards)
            rewards = (rewards - self.task2rew_ms[task].mean) / th.sqrt(self.task2rew_ms[task].var)
        
        # Train the critic
        critic_states_inputs, critic_individual_inputs = self._build_critic_inputs(batch, task)
        actions = actions.view(batch_size, -1, self.task2n_agents[task] * self.task2n_actions[task]) # (bs, T, n_agents * n_actions)
        q_taken1 = self.critic1(critic_states_inputs[:, :-1], actions[:, :-1], critic_individual_inputs[:, :-1], task) # (batch_size, T-1, n_agents, 1)
        q_taken2 = self.critic2(critic_states_inputs[:, :-1], actions[:, :-1], critic_individual_inputs[:, :-1], task)
       
        q_taken1 = q_taken1.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)
        q_taken2 = q_taken2.view(batch_size, -1, 1)

        # Use the target actor and target critic network to compute the target q
        self.target_mac.init_hidden(batch.batch_size, task)
        target_actions = []
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t, task) # (bs, n_agents, n_ac)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)[:, 1:]  # Concat over time, (bs, T-1, n_agents, n_ac)

        target_actions = target_actions.view(batch_size, -1, self.task2n_agents[task] * self.task2n_actions[task])
        target_vals1 = self.target_critic1(critic_states_inputs[:, 1:], target_actions.detach(), critic_individual_inputs[:, 1:], task)
        target_vals2 = self.target_critic2(critic_states_inputs[:, 1:], target_actions.detach(), critic_individual_inputs[:, 1:], task)
        target_vals = th.min(target_vals1, target_vals2)
        target_vals = target_vals.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)
        
        if self.main_args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.task2ret_ms[task].var) + self.task2ret_ms[task].mean

        targets = rewards.reshape(-1, 1) + self.main_args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals.reshape(-1, 1).detach()

        if self.main_args.standardise_returns:
            self.task2ret_ms[task].update(targets)
            targets = (targets - self.task2ret_ms[task].mean) / th.sqrt(self.task2ret_ms[task].var)
        
        td_error1 = (q_taken1.view(-1, 1) - targets.detach())
        td_error2 = (q_taken2.view(-1, 1) - targets.detach())
        masked_td_error1  = td_error1 * mask.reshape(-1, 1)
        masked_td_error2  = td_error2 * mask.reshape(-1, 1)
        td_loss = 0.5 * (masked_td_error1 ** 2).mean() + 0.5 * (masked_td_error2 ** 2).mean()
        critic_loss = td_loss

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.main_args.grad_norm_clip)
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()
        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()

        critic_log["td_error1_abs"] = masked_td_error1.abs().sum().item() / mask_elems
        critic_log["td_error2_abs"] = masked_td_error2.abs().sum().item() / mask_elems
        critic_log["q_taken1_mean"] = (q_taken1).sum().item() / mask_elems
        critic_log["q_taken2_mean"] = (q_taken2).sum().item() / mask_elems
        critic_log["target_mean"] = targets.sum().item() / mask_elems
        return critic_log
    

    def _build_critic_inputs(self, batch, task, t=None):
        
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        states_inputs = batch["state"][:, ts]
        individual_inputs = []
        if self.main_args.critic_individual_obs:
            individual_inputs.append(batch["obs"][:, ts])
        if self.main_args.critic_last_action:
            if t == 0:
                individual_inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                #assert all(batch["actions_onehot"][:, slice(t - 1, t)]==batch["actions_onehot"][:, t-1:t])
                individual_inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                        batch["actions_onehot"][:, :-1]], dim=1)
            individual_inputs.append(last_actions)
        if self.main_args.critic_agent_id:
            individual_inputs.append(th.eye(self.task2n_agents[task], device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, self.task2n_agents[task], -1))

        individual_inputs = th.cat(individual_inputs, dim=-1)
        return states_inputs, individual_inputs
    
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic1.cuda()
        self.target_critic1.cuda()
        self.critic2.cuda()
        self.target_critic2.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))