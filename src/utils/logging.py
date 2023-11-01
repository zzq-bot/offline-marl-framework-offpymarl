from collections import defaultdict
import logging
import numpy as np
from tensorboardX.writer import SummaryWriter

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        self.writer = SummaryWriter(logdir=directory_name)
        self.use_tb = True

    def setup_wandb(self, directory_name, project=None, name=None, run_id=None, entity=None, config=None):
        self.use_wandb = True
    
        import wandb
        self.wandb_run = (
            wandb.init(
                project=project,
                name=name,
                id=run_id,
                resume="allow",
                entity=entity,
                sync_tensorboard=True,
                config=config,  # type: ignore
            )
            if not wandb.run
            else wandb.run
        )
        self.wandb_run._label(repo="offpymarl")  # type: ignore
        self.writer = SummaryWriter(logdir=directory_name)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )
        
    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb or self.use_wandb:
            self.writer.add_scalar(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def log_histogram(self, key, value, t):
        self.writer.add_histogram(key, value, t)

    def log_embedding(self, key, value):
        self.writer.add_embedding(value, tag=key)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10}\t Episode: {:>10}\n".format(self.stats["episode"][-1][0], self.stats["episode"][-1][1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            #print(k)
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"

        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

