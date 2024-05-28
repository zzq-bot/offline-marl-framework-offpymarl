from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .null_learner import NullLearner
from .bc_learner import BCLearner
from .maddpg_learner import MADDPGLearner
from .matd3_learner import MATD3Learner
from .icq_learner import ICQLearner
from .itd3_learner import ITD3Learner
from .omar_learner import OMARLearner
from .cfcq_learner import CFCQLearner
from .omiga_learner import OMIGALearner

REGISTRY = {}

REGISTRY["bc_learner"] = BCLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["null_learner"] = NullLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["matd3_learner"] = MATD3Learner
REGISTRY["icq_learner"] = ICQLearner
REGISTRY["itd3_learner"] = ITD3Learner
REGISTRY["omar_learner"] = OMARLearner
REGISTRY["cfcq_learner"] = CFCQLearner
REGISTRY["omiga_learner"] = OMIGALearner

