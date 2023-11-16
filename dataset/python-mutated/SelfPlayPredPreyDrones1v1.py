import os
from gym_predprey_drones.envs.PredPreyDrones1v1 import PredPrey1v1PredDrone
from gym_predprey_drones.envs.PredPreyDrones1v1 import PredPrey1v1PreyDrone
from stable_baselines3.ppo.ppo import PPO as sb3PPO
from stable_baselines3.sac.sac import SAC as sb3SAC
import bach_utils.os as utos
import bach_utils.list as utlst
import bach_utils.sorting as utsrt
import bach_utils.sampling as utsmpl
OS = False

class SelfPlayEnvSB3:

    def __init__(self, algorithm_class, archive, sample_after_reset, sampling_parameters):
        if False:
            i = 10
            return i + 15
        self.algorithm_class = algorithm_class
        self.opponent_policy = None
        self.opponent_policy_name = None
        self.target_opponent_policy_name = None
        self._name = None
        self.archive = archive
        self.OS = OS
        self.sample_after_reset = sample_after_reset
        self.sampling_parameters = sampling_parameters
        if archive is None:
            self.OS = True
        self.states = None

    def set_target_opponent_policy_name(self, policy_name):
        if False:
            for i in range(10):
                print('nop')
        self.target_opponent_policy_name = policy_name

    def compute_action(self, obs):
        if False:
            return 10
        if self.opponent_policy is None:
            return self.action_space.sample()
        else:
            action = None
            if isinstance(self.opponent_policy, sb3PPO) or isinstance(self.opponent_policy, sb3SAC):
                (action, self.states) = self.opponent_policy.predict(obs, state=self.states)
            return action

    def _load_opponent(self, opponent_name):
        if False:
            return 10
        if opponent_name is not None and opponent_name != self.opponent_policy_name:
            self.opponent_policy_name = opponent_name
            if self.opponent_policy is not None:
                del self.opponent_policy
            if not self.OS:
                print(self.algorithm_class)
                self.opponent_policy = self.archive.load(name=opponent_name, env=self, algorithm_class=self.algorithm_class)
            if self.OS:
                self.opponent_policy = self.algorithm_class.load(opponent_name, env=self)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.states = None
        if self.sample_after_reset:
            print('Sample after reset the environment')
            opponent_selection = self.sampling_parameters['opponent_selection']
            sample_path = self.sampling_parameters['sample_path']
            startswith_keyword = 'history'
            randomly_reseed_sampling = self.sampling_parameters['randomly_reseed_sampling']
            sampled_opponent = None
            if not self.OS:
                archive = self.archive.get_sorted(opponent_selection)
                models_names = archive[0]
                sampled_opponent = utsmpl.sample_opponents(models_names, 1, selection=opponent_selection, sorted=True, randomly_reseed=randomly_reseed_sampling)[0]
            if self.OS:
                sampled_opponent = utsmpl.sample_opponents_os(sample_path, startswith_keyword, 1, selection=opponent_selection, randomly_reseed=randomly_reseed_sampling)[0]
            self.target_opponent_policy_name = sampled_opponent
        if self.OS:
            print(f'Reset, env name: {self._name}, OS, target_policy: {self.target_opponent_policy_name}')
        else:
            print(f'Reset, env name: {self._name}, archive_id: {self.archive.random_id}, target_policy: {self.target_opponent_policy_name}')
        self._load_opponent(self.target_opponent_policy_name)

class SelfPlayPredDroneEnv(SelfPlayEnvSB3, PredPrey1v1PredDrone):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', 'normal')
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)
        PredPrey1v1PredDrone.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.prey_policy = self

    def reset(self):
        if False:
            print('Hello World!')
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1PredDrone.reset(self)

class SelfPlayPreyDroneEnv(SelfPlayEnvSB3, PredPrey1v1PreyDrone):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        seed_val = kwargs.pop('seed_val')
        gui = kwargs.pop('gui', False)
        reward_type = kwargs.pop('reward_type', 'normal')
        SelfPlayEnvSB3.__init__(self, *args, **kwargs)
        PredPrey1v1PreyDrone.__init__(self, seed_val=seed_val, gui=gui, reward_type=reward_type)
        self.pred_policy = self

    def reset(self):
        if False:
            while True:
                i = 10
        SelfPlayEnvSB3.reset(self)
        return PredPrey1v1PreyDrone.reset(self)