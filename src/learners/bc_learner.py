from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
from components.standarize_stream import RunningMeanStd


class BCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        match self.args.optim_type.lower():
            case "rmsprop":
                self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.optimiser = Adam(params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.args.optim_type)
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"].type(th.long)
        
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
            
        #agent_outs[reshaped_avail_actions == 0] = -1e10

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        
        mac_out[avail_actions == 0] = -1e10
        #mac_out = F.softmax(mac_out, dim=-1) # get softmax policy

        bs, t, n_agents, ac_dim = mac_out.size()
        #print(mac_out.shape, actions.shape)
        loss = F.cross_entropy(mac_out.reshape(-1, ac_dim), actions.squeeze(-1).reshape(-1), reduction="sum")
        loss /= (mask.sum() * n_agents)

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(f"loss", loss.item(), t_env)
            self.logger.log_stat(f"grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
