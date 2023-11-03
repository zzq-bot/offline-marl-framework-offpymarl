import copy
import torch as th
import numpy as np
from torch.optim import Adam, RMSprop

from modules.critics.double import DoubleCritic
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets


class OMARLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.logger = logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.critic = DoubleCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
      

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        
  
        match self.args.optim_type.lower():
            case "rmsprop":
                self.agent_optimiser = RMSprop(params=self.agent_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
                self.critic_optimiser = RMSprop(params=self.critic_params, lr=self.args.critic_lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.agent_optimiser =  Adam(params=self.agent_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
                self.critic_optimiser =  Adam(params=self.critic_params, lr=args.critic_lr, weight_decay=getattr(args, "weight_decay", 0))
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
        critic_train_stats = self.train_critic(batch)
        
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        mask = mask.repeat(1, 1, self.n_agents).unsqueeze(-1) # (bs, seq_len-1, n_agents, 1)
        

        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_t - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # (bs, seq_len-1, n_agents, n_actions)

        mac_out[avail_actions == 0] =0
        pi_taken = th.gather(mac_out, dim=-1, index=actions)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        assert not th.isnan(log_pi_taken).any()
    
        critic_inputs = self.critic.build_inputs(batch)
        q_vals, _ = self.critic.forward(critic_inputs[:,:-1]) # (bs, max_t-1, n_agents, n_action)
        
        
        q_vals_taken = th.gather(q_vals, dim=-1, index=actions)
        baseline = th.sum(mac_out * q_vals, dim=-1, keepdim=True) # (bs, max_t-1, n_agents, 1)
        advantage = (q_vals_taken - baseline).detach()
        
        coma_loss = - (advantage * log_pi_taken * mask).sum() / mask.sum()
        
        #

        # #############omar####################
        # self.omar_mu = th.cuda.FloatTensor(bs,max_t-1,n_agents, 1).zero_() + action_dim/2
        # self.omar_sigma = th.cuda.FloatTensor(bs,max_t-1,n_agents, 1).zero_() + action_dim/4
        # repeat_avail_action = th.repeat_interleave(avail_actions.unsqueeze(-2),repeats=self.args.omar_num_samples,dim=-2)#bs,ts,na,nsample,ad
        # for iter_idx in range(self.args.omar_iters):
        #     dist = th.distributions.Normal(self.omar_mu, self.omar_sigma)

        #     cem_sampled_acs = dist.sample((self.args.omar_num_samples,)).permute(1,2,3,0,4).clamp(0, action_dim-1)
        #     cem_sampled_acs = th.div(cem_sampled_acs+0.5,1,rounding_mode='trunc').long()#discretize
        #     #bs,ts,na,nsample,1
        #     cem_sampled_avail = th.gather(repeat_avail_action,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1

        #     repeat_q_vals = th.repeat_interleave(q_vals.unsqueeze(-2),repeats=self.args.omar_num_samples,dim=-2)#bs,ts,na,nsample,ad
        #     all_pred_qvals = th.gather(repeat_q_vals,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1
        #     all_pred_qvals[cem_sampled_avail==0]=-1e10

        #     updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
        #     self.omar_mu = updated_mu#bs,ts,na,1
        #     updated_sigma = th.sqrt((th.mean((cem_sampled_acs - self.omar_mu.unsqueeze(-2)),-2) ** 2))
            
        #     self.omar_sigma = updated_sigma+0.0001#bs,ts,na,1

        # top_qvals, top_inds = th.topk(all_pred_qvals, 1, dim=-2)#bs,ts,na,1,1
        # top_acs = th.gather(cem_sampled_acs, -2, top_inds)#bs,ts,na,1,1
        # curr_pol_actions = mac_out.argmax(-1,keepdim=True)#bs,ts,na,1

        # cem_qvals = top_qvals.squeeze(-1)#bs,ts,na,1
        # pol_qvals = th.gather(q_vals, dim=3, index=curr_pol_actions)#bs,ts,na,1
        # pol_qvals = q_vals_taken
        # cem_acs = top_acs.squeeze(-1)#bs,ts,na,1
        # pol_acs = curr_pol_actions#bs,ts,na,1

        # candidate_qvals = th.cat([pol_qvals, cem_qvals], -1)#bs,ts,na,2
        # candidate_acs = th.cat([pol_acs, cem_acs], -1)#bs,ts,na,2

        # max_qvals, max_inds = th.max(candidate_qvals, -1, keepdim=True)#bs,ts,na,1

        # max_acs = th.gather(candidate_acs, -1, max_inds)#bs,ts,na,1
        # one_hot_max_acs = th.nn.functional.one_hot(max_acs,num_classes=action_dim).float()#bs,ts,na,ad

        max_q_acs = q_vals.argmax(-1, keepdim=True) # (bs, max_t-1, n_agents, 1)
        
        # one_hot_max_acs = th.nn.functional.one_hot(max_q_acs, num_classes=self.n_actions).float() # (bs, max_t-1, n_agents, n_actions)

        # omar_loss = th.nn.functional.mse_loss(mac_out.view(-1,action_dim), one_hot_max_acs.view(-1,action_dim).detach())
        omar_loss = th.nn.functional.cross_entropy(mac_out.view(-1, self.n_actions), max_q_acs.view(-1,))
        
        loss = (1-self.args.omar_coe) * coma_loss + self.args.omar_coe * omar_loss
    

       
        self.agent_optimiser.zero_grad()
        #self.critic_optimiser.zero_grad()
        loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        assert not th.isnan(actor_grad_norm).any()
        self.agent_optimiser.step()
        
        self.training_steps += 1

        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            
            for k, v in critic_train_stats.items():
                self.logger.log_stat(k, v, t_env)
            
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("omar_loss", omar_loss.item(), t_env)
            self.logger.log_stat("agent_lr", self.args.lr, t_env)
            
            self.logger.log_stat("actor_grad_norm", actor_grad_norm, t_env)
            self.logger.log_stat("pi_taken", (pi_taken * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
            
    def train_critic(self, batch):
        critic_log = {  "critic_loss":[],
                        "critic_grad_norm":[],  
                        "td_error_abs":[], 
                        "target_mean":[], 
                        "q_taken_mean":[]}
        bs = batch.batch_size
        max_t = batch.max_seq_length
        
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:]
        
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
            
        #build_target_q
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(bs)
            for t in range(max_t):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)  
            target_mac_out[avail_actions == 0] = 0
            target_policy_action = th.argmax(target_mac_out,dim=-1, keepdim=True)
                
            target_critic_inputs = self.target_critic.build_inputs(batch)
            target_q_vals_1,target_q_vals_2 = self.target_critic.forward(target_critic_inputs)
            target_q_vals_taken_1 = th.gather(target_q_vals_1, dim=-1, index=target_policy_action.long()) # (bs, max_t, n_agents, 1)
            target_q_vals_taken_2 = th.gather(target_q_vals_2, dim=-1, index=target_policy_action.long()) # (bs, max_t, n_agents, 1)
            
            
            repeat_rewards = th.repeat_interleave(rewards.unsqueeze(-2), repeats=self.n_agents, dim=-2) # (bs, max_t, n_agents, 1)    
            repeat_terminated = th.repeat_interleave(terminated.unsqueeze(-2), repeats=self.n_agents, dim=-2) # (bs, max_t, n_agents, 1)
            repeat_mask = th.repeat_interleave(mask.unsqueeze(-2), repeats=self.n_agents, dim=-2) # (bs, max_t, n_agents, 1)
            
            target_q_1 = build_td_lambda_targets(repeat_rewards, repeat_terminated, repeat_mask, target_q_vals_taken_1, self.n_agents, self.args.gamma, self.args.td_lambda).detach() # (bs, max_t, n_agents, 1)
            target_q_2 = build_td_lambda_targets(repeat_rewards, repeat_terminated, repeat_mask, target_q_vals_taken_2, self.n_agents, self.args.gamma, self.args.td_lambda).detach() # (bs, max_t, n_agents, 1)
            target_q = th.min(target_q_1, target_q_2).detach() # (bs, max_t, n_agents, 1)
            

        critic_inputs = self.critic.build_inputs(batch)
        # train critic, follow icq, is it reasonable?
        for t in range(max_t - 1):
            mask_t = repeat_mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals_1,q_vals_2 = self.critic.forward(critic_inputs[:, t:t+1])
            q_vals_taken_1 = th.gather(q_vals_1, index=actions[:,t:t+1], dim=-1)
            q_vals_taken_2 = th.gather(q_vals_2, index=actions[:,t:t+1], dim=-1)
            target_q_t = target_q[:, t:t+1].detach()
            q_err_1 = (q_vals_taken_1 - target_q_t) * mask_t
            q_err_2 = (q_vals_taken_2 - target_q_t) * mask_t
            td_loss = (q_err_1 ** 2).sum() / mask_t.sum() + (q_err_2 ** 2).sum() / mask_t.sum()
            
            cql_error_1 = th.logsumexp(q_vals_1, dim=-1, keepdim=True) - q_vals_taken_1
            cql_error_2 = th.logsumexp(q_vals_2, dim=-1, keepdim=True) - q_vals_taken_2
            cql_loss = (cql_error_1 * mask_t).sum() / mask_t.sum() + (cql_error_2 * mask_t).sum() / mask_t.sum()
            
            critic_loss = td_loss + self.args.cql_alpha * cql_loss
            
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            assert not th.isnan(grad_norm).any()
            self.critic_optimiser.step()
            
            critic_log["critic_loss"].append(critic_loss.item())
            critic_log["critic_grad_norm"].append(grad_norm.item())
            mask_elems = mask_t.sum().item()
            critic_log["td_error_abs"].append((q_err_1.abs().sum().item() / mask_elems))
            critic_log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            critic_log["q_taken_mean"].append((q_vals_taken_1 * mask_t).sum().item() / mask_elems)
        for k in critic_log.keys():
            critic_log[k] = np.mean(critic_log[k])
        return critic_log
    
    
    # def compute_softmax_acs(self, q_vals, acs):
    #     max_q_vals = th.max(q_vals, -2, keepdim=True)[0]#bs,ts,na,1,1
    #     norm_q_vals = q_vals - max_q_vals
    #     e_beta_normQ = th.exp(norm_q_vals)#bs,ts,na,nsample,1
    #     a_mult_e = acs * e_beta_normQ#bs,ts,na,nsample,1
    #     numerators = a_mult_e
    #     denominators = e_beta_normQ

    #     sum_numerators = th.sum(numerators, -2)
    #     sum_denominators = th.sum(denominators, -2)

    #     softmax_acs = sum_numerators / sum_denominators#bs,ts,na,1

    #     return softmax_acs
    
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        assert tau <= 1
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
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self._update_targets()