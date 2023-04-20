from components.episode_buffer import EpisodeBatch


class NullLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
       pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass
