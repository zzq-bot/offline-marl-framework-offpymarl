from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
from components.standarize_stream import RunningMeanStd


class MTBCLearner:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer
        self.task2input_shape_info = mac.task2input_shape_info

        self.params = list(mac.parameters())

        match self.main_args.optim_type.lower():
            case "rmsprop":
                self.optimiser = RMSprop(params=self.params, lr=self.main_args.lr, alpha=self.main_args.optim_alpha, eps=self.main_args.optim_eps, weight_decay=self.main_args.weight_decay)
            case "adam":
                self.optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.args.optim_type)
        
        self.task2train_info = {}
        for task, task_args in self.task2args.items():
            self.task2train_info[task] = {
                "log_stats_t": -task_args.learner_log_interval - 1,
                "training_steps": 0
            }
        self.last_target_update_episode = 0
        self.total_training_steps = 0

        device = "cuda" if main_args.use_cuda else "cpu"
        if self.main_args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.main_args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task:str):
        # Get the relevant quantities
        actions = batch["actions"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"].type(th.long)
        
        if self.main_args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
            
        #agent_outs[reshaped_avail_actions == 0] = -1e10

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task=task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)
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
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm.item(), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
