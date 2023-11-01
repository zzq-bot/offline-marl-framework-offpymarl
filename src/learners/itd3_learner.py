import copy
from components.episode_buffer import EpisodeBatch
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from modules.critics.itd3 import ITD3Critic
from components.standarize_stream import RunningMeanStd



class ITD3Learner:
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
        self.last_target_update_step = 0
        self.last_target_update_episode = 0
        
        self.log_actor = {"actor_loss":[], "actor_grad_norm":[]}
        if "bc" in self.args.name:
            self.log_actor["bc_loss"] = []
            self.log_actor["td3_loss"] = []

        if "omar" in self.args.name:
            self.omar_coe = args.omar_coe
            self.omar_iters = args.omar_iters
            self.omar_num_samples = args.omar_num_samples
            #self.omar_num_elites = args.omar_num_elites
            self.init_omar_mu = args.init_omar_mu
            self.init_omar_sigma = args.init_omar_sigma
            self.log_actor["omar_loss"] = []
            self.log_actor["td3_loss"] = []
    
        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
    

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        critic_log, top_acs = self.train_critic(batch)
        
        if (self.training_steps + 1) % self.actor_freq == 0:
            batch_size = batch.batch_size
        
            critic_inputs = self._build_critic_inputs(batch)
            avail_actions = batch["avail_actions"][:, :-1]
            actions_4bc = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
            mask = 1 - terminated
            # Train the actor
            self.mac.init_hidden(batch_size)
            self.critic1.init_hidden(batch.batch_size)
            pis = []
            actions = []
            q = []
            for t in range(batch.max_seq_length-1):
                pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1) # (batch_size, 1, n_agents, n_acs)
                pis.append(pi)
                actions.append(gumbel_softmax(pi, hard=True)) # (batch_size, 1, n_agents, n_acs)
                q.append(self.critic1(critic_inputs[:, t], actions[t].squeeze(1)))

            actions = th.cat(actions, dim=1) # (batch_size, T-1, n_agents, n_acs)
            q = th.stack(q, dim=1).reshape(-1, 1) # (bs * (T-1) * n_agents, 1)

            if "bc" in self.args.name: # matd3+bc
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis_mask = mask.expand_as(pis)
                pis = pis.reshape(-1, self.n_actions)
                bc_loss = F.cross_entropy(pis, actions_4bc.reshape(-1), reduction="sum")
                bc_loss = bc_loss/(pis_mask.sum())
                
                mask = mask.reshape(-1, 1)
                #lmbda = self.args.td3_alpha / ((q * mask).sum()/mask.sum()).abs().mean().detach()
                lmbda = self.args.td3_alpha / ((q * mask).abs().sum().detach() / mask.sum()) 
                td3_loss = - lmbda * (q * mask).mean() 
                
                actor_loss = td3_loss + bc_loss
                
            elif "omar" in self.args.name:
                raise NotImplementedError("Omar is not implemented yet")
                # Omar
                # self.omar_mu = th.zeros((batch_size, batch.max_seq_length-1, self.n_agents, self.n_actions)) + self.init_omar_mu
                # self.omar_sigma = th.zeros((batch_size, batch.max_seq_length-1, self.n_agents, self.n_actions)) + self.init_omar_sigma + 1e-5
                # self.omar_mu = self.omar_mu.to(batch.device)
                # self.omar_sigma = self.omar_sigma.to(batch.device)
                # formatted_critic_inputs = critic_inputs.unsqueeze(0).repeat(self.omar_num_samples, 1, 1, 1, 1).\
                #     view(self.omar_num_samples * batch_size, batch.max_seq_length, self.n_agents, -1)[:, :-1]
                # formatted_avail_actions = avail_actions.unsqueeze(0).repeat(self.omar_num_samples, 1, 1, 1, 1).\
                #     view(self.omar_num_samples * batch_size, -1, self.n_agents, self.n_actions)
                # for iter_idx in range(self.omar_iters):
                #     # print(th.all(self.omar_mu>=0), th.all(self.omar_sigma >= 0))
                #     # assert th.all(self.omar_sigma >= 0)
                #     # self.omar_sigma = th.zeros((batch_size, batch.max_seq_length-1, self.n_agents, self.n_actions)) + self.init_omar_sigma
                #     # self.omar_sigma = self.omar_sigma.to(batch.device)
                #     dist = th.distributions.Normal(self.omar_mu, self.omar_sigma)
                #     cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach() # (samples, bs, T-1, n_agents, n_actions)
                #     cem_sampled_acs = cem_sampled_acs.view(self.omar_num_samples * batch_size, -1, self.n_agents, self.n_actions)
                #     cem_sampled_acs[formatted_avail_actions == 0] = -1e10
                #     discrete_cem_sampled_acs = F.one_hot(cem_sampled_acs.argmax(dim=-1))  # (samples*bs, T-1, n_agents, n_actions)
                #     # print(discrete_cem_sampled_acs.shape, formatted_critic_inputs.shape)
                #     all_pred_qvals = self.critic1(formatted_critic_inputs, discrete_cem_sampled_acs).\
                #         view(self.omar_num_samples, batch_size, -1, self.n_agents, 1)
                #     discrete_cem_sampled_acs = discrete_cem_sampled_acs.\
                #         view(self.omar_num_samples, batch_size, -1, self.n_agents, self.n_actions)
            
                #     # self.omar_mu = self._compute_softmax_acs(all_pred_qvals, discrete_cem_sampled_acs)
                #     cem_sampled_acs = cem_sampled_acs.view(self.omar_num_samples, batch_size, -1, self.n_agents, self.n_actions)
                #     self.omar_mu = self._compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
                #     # self.omar_sigma  = th.sqrt(th.mean((discrete_cem_sampled_acs - self.omar_mu.unsqueeze(0)) ** 2, 0)) + 1e-5
                #     self.omar_sigma  = th.sqrt(th.mean((cem_sampled_acs - self.omar_mu.unsqueeze(0)) ** 2, 0)) + 1e-5

                # top_qvals, top_inds = th.topk(all_pred_qvals, 1, dim=0)
                # top_inds = top_inds # (1, bs, T-1, n_agents, 1)
                # top_acs = th.gather(discrete_cem_sampled_acs, dim=0, index=top_inds).squeeze(0).argmax(dim=-1) # (bs, T-1, n_agents)
                
                # get top_acs from q_values
                # assert top_acs is not None
                # pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                # pis_mask = mask.expand_as(pis)
                # pis = pis.reshape(-1, self.n_actions)
                # omar_loss = F.cross_entropy(pis, top_acs.reshape(-1).detach(), reduction="mean")
                # omar_loss = omar_loss / (pis_mask.sum())

                # mask = mask.reshape(-1, 1)
                # td3_loss =  -(q * mask).mean() 

                # actor_loss = (1 - self.omar_coe) * td3_loss + self.omar_coe * omar_loss 
            else: 
                pis = th.cat(pis, dim=1) # (bs, (T-1), n_agents, n_actions)
                pis[pis==-1e10] = 0
                masked_pis = pis * mask.expand_as(pis)
                masked_pis = masked_pis.reshape(-1, 1) # (bs * (T-1) * n_agents, n_actions)

                mask = mask.reshape(-1, 1)
                actor_loss = - (q * mask).mean() + self.args.reg * ( masked_pis ** 2).mean()

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
            # if "omar" in self.args.name:
            #     #raise NotImplementedError("Omar is not implemented yet")
            #     self.log_actor["omar_loss"].append(omar_loss.item())
            #     self.log_actor["td3_loss"].append(td3_loss.item())

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

    def train_critic(self, batch: EpisodeBatch):
        critic_log = {}
        top_acs = None
        batch_size = batch.batch_size

        rewards = batch["reward"][:, :-1] # (bs, T-1, 1)
        actions = batch["actions_onehot"]
        avail_actions = batch["avail_actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        rewards = rewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        terminated = terminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1) # (bs, T-1, n_agents, 1)
        mask = 1 - terminated

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
        # Train the critic
        # Compute current 
        critic_inputs = self._build_critic_inputs(batch) # (batch_size, T, n_agents, x)
        # (batch_size, T, n_agents, n_actions)
        self.critic1.init_hidden(batch_size)
        self.critic2.init_hidden(batch_size)
        q_taken1 = []
        q_taken2 = []
        for t in range(batch.max_seq_length-1):
            q_taken1.append(self.critic1(critic_inputs[:, t], actions[:, t]))
            q_taken2.append(self.critic2(critic_inputs[:, t], actions[:, t]))
        q_taken1 = th.stack(q_taken1, dim=1) # (batch_size, T-1, n_agents, 1)
        q_taken2 = th.stack(q_taken2, dim=1)


        # Use the target actor and target critic network to compute the target q
        self.target_mac.init_hidden(batch.batch_size)
        self.target_critic1.init_hidden(batch_size)
        self.target_critic2.init_hidden(batch_size)
        #target_actions = []
        target_vals1 = []
        target_vals2 = []
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.target_actions(batch, t) # (bs, n_agents, n_ac)
            target_vals1.append(self.target_critic1(critic_inputs[:, t], agent_target_outs.detach()))
            target_vals2.append(self.target_critic2(critic_inputs[:, t], agent_target_outs.detach()))
        target_vals1 = th.stack(target_vals1[1:], dim=1)
        target_vals2 = th.stack(target_vals2[1:], dim=1)
        target_vals = th.min(target_vals1, target_vals2)
        target_vals = target_vals.reshape(-1, 1) # (batch_size*(T-1)*n_agents, 1)
        
        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_vals

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
        
        td_error1 = (q_taken1.reshape(-1, 1) - targets.detach())
        td_error2 = (q_taken2.reshape(-1, 1) - targets.detach())
        masked_td_error1  = td_error1 * mask.reshape(-1, 1)
        masked_td_error2  = td_error2 * mask.reshape(-1, 1)
        td_loss = 0.5 * (masked_td_error1 ** 2).mean() + 0.5 * (masked_td_error2 ** 2).mean()

        # if ("cql" in self.args.name or "omar" in self.args.name) and getattr(self.args, "cql_type", "vanilla")=="vanilla":
        #     # get over actions
        #     assert self.args.critic_rnn is False
        #     consq1 = []
        #     consq2 = []
        #     for i in range(self.n_actions):
        #         consq1_i, consq2_i = [], []
        #         self.critic1.init_hidden(batch.batch_size)
        #         self.critic2.init_hidden(batch.batch_size)
        #         tmp_action = th.zeros(batch.batch_size, self.n_agents, self.n_actions).to(batch.device)
        #         tmp_action[:, :, i] = 1
        #         for t in range(batch.max_seq_length-1):
        #             consq1_i.append(self.critic1(critic_inputs[:, t], tmp_action))
        #             consq2_i.append(self.critic2(critic_inputs[:, t], tmp_action))
        #         consq1_i = th.stack(consq1_i, dim=1) # (bs, T-1, n_agents, 1)
        #         consq2_i = th.stack(consq2_i, dim=1) # (bs, T-1, n_agents, 1)
        #         consq1.append(consq1_i)
        #         consq2.append(consq2_i)
        #     consq1 = th.cat(consq1, dim=-1)
        #     consq2 = th.cat(consq2, dim=-1)
        #     consq1[avail_actions == 0] = 0
        #     consq2[avail_actions == 0] = 0
        #     #q_values = th.min(consq1, consq2) # (bs, T-1, n_agents, n_actions)
        #     top_acs = consq1.argmax(dim=-1) # (bs, T-1, n_agents)
        #     cql_loss1 = ((th.logsumexp(consq1, dim=-1).reshape(-1, 1) - q_taken1.reshape(-1, 1)) * mask.reshape(-1, 1)).sum() / mask.sum()
        #     cql_loss2 = ((th.logsumexp(consq2, dim=-1).reshape(-1, 1) - q_taken2.reshape(-1, 1)) * mask.reshape(-1, 1)).sum() / mask.sum()
        #     cql_loss = (cql_loss1 + cql_loss2) / 2
        #     critic_loss = self.args.cql_alpha * cql_loss + td_loss
        # elif "cql" in self.args.name:
        #     raise NotImplementedError("not implenmeneted so far")
        # else:   
        critic_loss = td_loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mask_elems = mask.sum().item()

        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()
        
        critic_log["td_error1_abs"] = masked_td_error1.abs().sum().item() / mask_elems
        critic_log["td_error2_abs"] = masked_td_error2.abs().sum().item() / mask_elems
        critic_log["q_taken1_mean"] = (q_taken1).sum().item() / mask_elems
        critic_log["q_taken2_mean"] = (q_taken2).sum().item() / mask_elems
        critic_log["target_mean"] = targets.sum().item() / mask_elems
        return critic_log, top_acs
        
    def _build_critic_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []    
        inputs.append(batch["obs"][:, ts])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                #assert all(batch["actions_onehot"][:, slice(t - 1, t)]==batch["actions_onehot"][:, t-1:t])
                inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                        batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        if self.args.obs_agent_id:
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
    
    def _compute_softmax_acs(self, q_vals, acs):
        # (num_sampled, bs, T-1, n_agents, 1 or n_actions)
        max_q_vals = th.max(q_vals, 0, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(norm_q_vals)
        a_mult_e = acs * e_beta_normQ
        numerators = a_mult_e
        denominators = e_beta_normQ
        sum_numerators = th.sum(numerators, 0)
        sum_denominators = th.sum(denominators, 0)

        softmax_acs = sum_numerators / sum_denominators

        return softmax_acs
    
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
