from gym_pz_predprey.envs.PZPredPrey import PZPredPreyPred
from gym_pz_predprey.envs.PZPredPrey import PZPredPreyPrey
from bach_utils.SelfPlay import SelfPlayEnvSB3

class SelfPlayPZPredEnv(SelfPlayEnvSB3, PZPredPreyPred):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', None)
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)
        PZPredPreyPred.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.prey_policy = self

    def reset(self):
        if False:
            print('Hello World!')
        SelfPlayEnvSB3.reset(self)
        return PZPredPreyPred.reset(self)

class SelfPlayPZPreyEnv(SelfPlayEnvSB3, PZPredPreyPrey):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', None)
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)
        PZPredPreyPrey.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.pred_policy = self

    def reset(self):
        if False:
            while True:
                i = 10
        SelfPlayEnvSB3.reset(self)
        return PZPredPreyPrey.reset(self)