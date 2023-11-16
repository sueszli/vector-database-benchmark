import numpy as np

def add_RewardUncertaintyEnvClassWrapper(EnvClass, reward_uncertainty_std, reward_uncertainty_mean=0.0):
    if False:
        while True:
            i = 10

    class RewardUncertaintyEnvClassWrapper(EnvClass):

        def step(self, action):
            if False:
                return 10
            (observations, rewards, done, info) = super().step(action)
            return (observations, self.reward_wrapper(rewards), done, info)

        def reward_wrapper(self, reward_dict):
            if False:
                while True:
                    i = 10
            for k in reward_dict.keys():
                reward_dict[k] += np.random.normal(loc=reward_uncertainty_mean, scale=reward_uncertainty_std, size=())
            return reward_dict
    return RewardUncertaintyEnvClassWrapper