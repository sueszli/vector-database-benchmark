import os
import ray
from ray import tune
import gym
from ray.tune.logger import pretty_print
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
import torch
import numpy as np
OBS = 'full'
ACT = 'vel'
ENV = 'PredPrey-Pred-v0'
WORKERS = 6
ALGO = 'PPO'
SEED_VALUE = 3
TRAINING_ITERATION = 1000

def make_deterministic(seed):
    if False:
        while True:
            i = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True

def create_environment(config):
    if False:
        print('Hello World!')
    import gym_predprey
    env = gym.make('PredPrey-Pred-v0')
    return env

class SelfPlayPolicies:

    def __init__(self, num_policies=2, initialization_policy=RandomPolicy, keys=None):
        if False:
            for i in range(10):
                print('nop')
        self.num_policies = num_policies
        if keys is None:
            self.keys = [i for i in range(num_policies)]
        else:
            if num_policies != len(keys):
                print('Number of policies is not equal to number of keys provideed to map the policy to the dictionary')
                raise ValueError
            self.keys = keys
        self.policies = {self.keys[i]: [initialization_policy] for i in range(num_policies)}

    def sample(self, num_sampled_policies=1):
        if False:
            return 10
        policies = {}
        for i in range(self.num_policies):
            key = self.keys[i]
            policies[key] = np.random.choice(self.policies[key], num_sampled_policies)
        return policies

    def store(self, policies):
        if False:
            print('Hello World!')
        for i in range(self.num_policies):
            key = self.keys[i]
            self.policies[key].append(policies[key])
if __name__ == '__main__':
    if torch.cuda.is_available():
        print('## CUDA available')
        print(f'Current device: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('## CUDA not available')
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + '/results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime('%m.%d.%Y_%H.%M.%S')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir + '/')
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    register_env(ENV, create_environment)
    ppo_default_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config = {**ppo_default_config, **{}}
    config = {'env': ENV, 'num_workers': 0 + WORKERS, 'num_gpus': 1 if torch.cuda.is_available() else 0, 'batch_mode': 'complete_episodes', 'seed': SEED_VALUE, 'framework': 'torch'}
    print(pretty_print(config))
    stop = {'training_iteration': TRAINING_ITERATION}
    results = tune.run(ALGO, stop=stop, config={**config, **ppo_config}, log_to_file=['logs_out.txt', 'logs_err.txt'], checkpoint_freq=5, checkpoint_at_end=True, local_dir=checkpoint_dir)
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean', mode='max'), metric='episode_reward_mean')
    with open(checkpoint_dir + '/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])
    ray.shutdown()