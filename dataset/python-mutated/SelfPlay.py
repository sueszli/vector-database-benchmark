from bach_utils.logger import get_logger
clilog = get_logger()
import os
from stable_baselines3.ppo.ppo import PPO as sb3PPO
from stable_baselines3.sac.sac import SAC as sb3SAC
import bach_utils.sampling as utsmpl
OS = False

class SelfPlayEnvSB3:

    def __init__(self, algorithm_class, archive, sample_after_reset, sampling_parameters):
        if False:
            i = 10
            return i + 15
        self.opponent_algorithm_class = algorithm_class
        self.opponent_policy = None
        self.opponent_policy_name = None
        self.target_opponent_policy_name = None
        self._name = None
        self.archive = archive
        self.OS = OS
        self.sample_after_reset = sample_after_reset
        self.sampling_parameters = sampling_parameters
        self.reset_counter = 0
        if archive is None:
            self.OS = True
        self.states = None

    def set_target_opponent_policy_name(self, policy_name):
        if False:
            print('Hello World!')
        self.target_opponent_policy_name = policy_name

    def compute_action(self, obs):
        if False:
            return 10
        if self.opponent_policy is None:
            return self.action_space.sample()
        else:
            action = None
            deterministic = None
            if self._name in ['Training', 'Evaluation']:
                deterministic = False
            else:
                deterministic = True
            if isinstance(self.opponent_policy, sb3PPO) or isinstance(self.opponent_policy, sb3SAC):
                (action, self.states) = self.opponent_policy.predict(obs, state=self.states, deterministic=deterministic)
            return action

    def _load_opponent(self, opponent_name):
        if False:
            return 10
        if opponent_name is not None:
            if 'Training' in self._name:
                clilog.debug(f'Add frequency +1 for {opponent_name}')
                self.archive.add_freq(opponent_name, 1)
            if opponent_name != self.opponent_policy_name:
                self.opponent_policy_name = opponent_name
                if self.opponent_policy is not None:
                    del self.opponent_policy
                if not self.OS:
                    self.opponent_policy = self.archive.load(name=opponent_name, env=self, algorithm_class=self.opponent_algorithm_class)
                if self.OS:
                    self.opponent_policy = self.opponent_algorithm_class.load(opponent_name, env=self)
                clilog.debug(f'loading opponent model: {opponent_name}, {self.opponent_policy}, {self}')

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.states = None
        if self.sample_after_reset:
            clilog.debug('Sample after reset the environment')
            opponent_selection = self.sampling_parameters['opponent_selection']
            sample_path = self.sampling_parameters['sample_path']
            startswith_keyword = 'history'
            randomly_reseed_sampling = self.sampling_parameters['randomly_reseed_sampling']
            sampled_opponent = None
            if not self.OS:
                archive = self.archive.get_sorted(opponent_selection)
                models_names = archive[0]
                sampled_opponent = utsmpl.sample_opponents(models_names, 1, selection=opponent_selection, sorted=True, randomly_reseed=randomly_reseed_sampling, delta=self.archive.delta, idx=self.reset_counter)[0]
            if self.OS:
                sampled_opponent = utsmpl.sample_opponents_os(sample_path, startswith_keyword, 1, selection=opponent_selection, randomly_reseed=randomly_reseed_sampling, delta=self.archive.delta, idx=self.reset_counter)[0]
            self.target_opponent_policy_name = sampled_opponent
        if self.OS:
            clilog.debug(f'Reset, env name: {self._name}, OS, target_policy: {self.target_opponent_policy_name} ({str(self.opponent_algorithm_class)})')
        else:
            clilog.debug(f'Reset, env name: {self._name}, archive_id: {self.archive.random_id}, target_policy: {self.target_opponent_policy_name} ({str(self.opponent_algorithm_class)})')
        self._load_opponent(self.target_opponent_policy_name)
        self.reset_counter += 1