import copy
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd



class MATD3Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.actor_freq = args.actor_freq
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic1 = critic_registry[args.critic_type](scheme, args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2 = critic_registry[args.critic_type](scheme, args)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())

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
        #self.last_target_update_step = 0
        self.last_target_update_episode = 0
        
        self.log_actor = {"actor_loss":[], "actor_grad_norm":[]}
        if "bc" in self.args.name:
            self.log_actor["bc_loss"] = []
            self.log_actor["td3_loss"] = []

        # if "cql" in self.args.name:
        #     self.cql_alpha = self.args.cql_alpha
        #     self.cql_temperature = self.args.cql_temperature
        #     self.num_repeats = self.args.num_repeats

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
    

    def train(self, batch, t_env: int, episode_num: int):
        critic_log = self.train_critic(batch)
        
        if (self.training_steps + 1) % self.actor_freq == 0:
            batch_size = batch.batch_size
        
            critic_inputs = self._build_critic_inputs(batch)
            actions_4bc = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
            mask = 1 - terminated
            # Train the actor
            self.mac.init_hidden(batch_size)
            pis = []
            actions = []
            for t in range(batch.max_seq_length-1):
                pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1) # (batch_size, 1, n_agents, n_acs)
                pis.append(pi) # avail_actions have been masked already
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

            q = self.critic1(critic_inputs[:, :-1], new_actions)
            q = q.reshape(-1, 1) # (bs * (T-1) * n_agents, 1)

            if "bc" in self.args.name: # matd3+bc
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis_mask = mask.expand_as(pis)
                pis = pis.reshape(-1, self.n_actions)
                #pis = pis * pis_mask.reshape(-1, self.n_actions)
                bc_loss = F.cross_entropy(pis, actions_4bc.reshape(-1), reduction="sum")
                bc_loss = bc_loss/(pis_mask.sum())
                
                mask = mask.reshape(-1, 1)
                #lmbda = self.args.td3_alpha / ((q * mask).sum()/mask.sum()).abs().mean().detach()
                lmbda = self.args.td3_alpha / ((q * mask).abs().sum().detach() / mask.sum()) 
                td3_loss = - lmbda * (q * mask).mean() 
                
                actor_loss = td3_loss + bc_loss
            else: 
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis[pis==-1e10] = 0
                masked_pis = pis * mask.expand_as(pis)
                masked_pis = masked_pis.reshape(-1, 1) # (bs * (T-1) * n_agents, n_actions)

                mask = mask.reshape(-1, 1)
                actor_loss = - (q * mask).mean() + self.args.reg * ( masked_pis ** 2).mean()
                #actor_loss = bc_loss
            # Optimise agents
            self.agent_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

            if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)
            # save record
            self.log_actor["actor_loss"].append(actor_loss.item())
            self.log_actor["actor_grad_norm"].append(actor_grad_norm.item())
            if "bc" in self.args.name:
                self.log_actor["bc_loss"].append(bc_loss.item())
                self.log_actor["td3_loss"].append(td3_loss.item())

        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in critic_log.items():
                self.logger.log_stat(k, v, t_env)
            if len(self.log_actor["actor_loss"]) > 0:
                ts = len(self.log_actor["actor_loss"])
                for k, v in self.log_actor.items():
                    self.logger.log_stat(k, sum(v)/ts, t_env)
                    self.log_actor[k].clear()

            self.log_stats_t = t_env

    def train_critic(self, batch):
        critic_log = {}
        batch_size = batch.batch_size

        
        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        actions = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        mask = 1 - terminated

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
        # Train the critic
        critic_inputs = self._build_critic_inputs(batch)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        q_taken1 = self.critic1(critic_inputs[:, :-1], actions[:, :-1].detach()) # (batch_size, T-1, n_agents, 1)
        q_taken2 = self.critic2(critic_inputs[:, :-1], actions[:, :-1].detach())
       
        q_taken1 = q_taken1.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)
        q_taken2 = q_taken2.view(batch_size, -1, 1)

        # Use the target actor and target critic network to compute the target q
        self.target_mac.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t) # (bs, n_agents, n_ac)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)[:, 1:]  # Concat over time, (bs, T-1, n_agents, n_ac)

        target_actions = target_actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        target_vals1 = self.target_critic1(critic_inputs[:, 1:], target_actions.detach())
        target_vals2 = self.target_critic2(critic_inputs[:, 1:], target_actions.detach())
        target_vals = th.min(target_vals1, target_vals2)
        target_vals = target_vals.view(batch_size, -1, 1) # (batch_size, (T-1)*n_agents, 1)
        
        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals.reshape(-1, 1).detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        
        td_error1 = (q_taken1.view(-1, 1) - targets.detach())
        td_error2 = (q_taken2.view(-1, 1) - targets.detach())
        masked_td_error1  = td_error1 * mask.reshape(-1, 1)
        masked_td_error2  = td_error2 * mask.reshape(-1, 1)
        td_loss = 0.5 * (masked_td_error1 ** 2).mean() + 0.5 * (masked_td_error2 ** 2).mean()


        # if "cql" in self.args.name:
        #     raise NotImplementedError("bad performance, to be improved")
        #     # critic_inputs.shape = (bs, T-1, n_agents, x)
        #     Tm1 = critic_inputs.shape[1]-1
        #     formatted_critic_inputs = critic_inputs.unsqueeze(3).repeat(1, 1, 1, self.num_repeats, 1)
            
        #     #### q_rand (bs*(T-1)*n_agents, num_repeats 1)
        #     random_actions = F.one_hot(th.randint(low=0, high=self.n_actions, size=(batch_size, Tm1, self.num_repeats, self.n_agents), device=batch.device))
        #     random_actions = random_actions.view(batch_size, Tm1, self.num_repeats, 1, self.n_agents*self.n_actions).expand(-1, -1, -1, self.n_agents, -1)
        #     random_actions = random_actions.transpose(2, 3)
        #     #random_log_prob = np.log(1 / self.n_actions * self.n_actions) # = 0
        #     random_Q1 = self.critic1(formatted_critic_inputs[:, :-1], random_actions) #- random_log_prob
        #     random_Q2 = self.critic2(formatted_critic_inputs[:, :-1], random_actions) #- random_log_prob
        #     random_Q1 = random_Q1.reshape(-1, self.num_repeats, 1)
        #     random_Q2 = random_Q2.reshape(-1, self.num_repeats, 1)
                
            
        #     #### q_cur

        #     cur_actions, cur_log_probs = [], []
        #     with th.no_grad():
        #         self.mac.init_hidden(batch_size, self.num_repeats)
        #         for t in range(batch.max_seq_length-1):
        #             cur_action, cur_log_prob = self.mac.get_repeat_actions(batch, t, self.num_repeats)
        #             cur_actions.append(cur_action)
        #             cur_log_probs.append(cur_log_prob)
        #         cur_actions = th.stack(cur_actions, dim=1) # (bs, T-1, n_agents, num_repeats, n_actions)
        #         cur_log_probs = th.stack(cur_log_probs, dim=1).unsqueeze(-1) # (bs, T-1, n_agents, num_repeats, 1)

        #     cur_actions = cur_actions.transpose(2, 3).contiguous().view(batch_size, Tm1, self.num_repeats, 1, self.n_agents*self.n_actions)
        #     cur_actions = cur_actions.expand(-1, -1, -1, self.n_agents, -1).transpose(2, 3)
        #     cur_Q1 = self.critic1(formatted_critic_inputs[:, :-1], cur_actions.detach())# - cur_log_probs.detach()
        #     cur_Q2 = self.critic2(formatted_critic_inputs[:, :-1], cur_actions.detach())# - cur_log_probs.detach()
        #     cur_Q1 = cur_Q1.reshape(-1, self.num_repeats, 1)
        #     cur_Q2 = cur_Q2.reshape(-1, self.num_repeats, 1)
            
        #     #### q_nxt
        #     nxt_actions, nxt_log_probs = [], []
        #     with th.no_grad():
        #         self.mac.init_hidden(batch_size, self.num_repeats)
        #         for t in range(batch.max_seq_length):
        #             nxt_action, nxt_log_prob = self.mac.get_repeat_actions(batch, t, self.num_repeats)
        #             nxt_actions.append(nxt_action)
        #             nxt_log_probs.append(nxt_log_prob)
        #         nxt_actions = th.stack(nxt_actions, dim=1)[:, 1:] # (bs, T-1, n_agents, num_repeats, n_actions)
        #         nxt_log_probs = th.stack(nxt_log_probs, dim=1).unsqueeze(-1)[:, 1:] # (bs, T-1, n_agents, num_repeats, 1)
        #     nxt_actions = nxt_actions.transpose(2, 3).contiguous().view(batch_size, Tm1, self.num_repeats, 1, self.n_agents*self.n_actions)
        #     nxt_actions = nxt_actions.expand(-1, -1, -1, self.n_agents, -1).transpose(2, 3)
        #     nxt_Q1 = self.critic1(formatted_critic_inputs[:, 1:], nxt_actions.detach())# - nxt_log_probs.detach()
        #     nxt_Q2 = self.critic2(formatted_critic_inputs[:, 1:], nxt_actions.detach())# - nxt_log_probs.detach()
        #     nxt_Q1 = nxt_Q1.reshape(-1, self.num_repeats, 1)
        #     nxt_Q2 = nxt_Q2.reshape(-1, self.num_repeats, 1)
            
        #     cat_Q1 = th.cat([random_Q1, cur_Q1, nxt_Q1], dim=1)
        #     cat_Q2 = th.cat([random_Q2, cur_Q2, nxt_Q2], dim=1)
        #     cat_q1_vals = th.logsumexp(cat_Q1 / self.cql_temperature, dim=1) * self.cql_temperature
        #     cat_q2_vals = th.logsumexp(cat_Q2 / self.cql_temperature, dim=1) * self.cql_temperature
        #     masked_cql_loss1 = ((cat_q1_vals-q_taken1.reshape(-1, 1)) * mask.reshape(-1, 1)).sum() / mask.sum()
        #     masked_cql_loss2 = ((cat_q2_vals-q_taken2.reshape(-1, 1)) * mask.reshape(-1, 1)).sum() / mask.sum()

        #     cql_loss = (masked_cql_loss1 + masked_cql_loss2) / 2

        #     critic_loss = td_loss + self.cql_alpha * cql_loss
        # else:
        critic_loss = td_loss

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()
        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()
        # if "cql" in self.args.name:
        #     raise NotImplementedError()
        #     critic_log["td_loss"] = td_loss.item()
        #     critic_log["cql_loss"] = cql_loss.item()

        critic_log["td_error1_abs"] = masked_td_error1.abs().sum().item() / mask_elems
        critic_log["td_error2_abs"] = masked_td_error2.abs().sum().item() / mask_elems
        critic_log["q_taken1_mean"] = (q_taken1).sum().item() / mask_elems
        critic_log["q_taken2_mean"] = (q_taken2).sum().item() / mask_elems
        critic_log["target_mean"] = targets.sum().item() / mask_elems
        return critic_log

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
