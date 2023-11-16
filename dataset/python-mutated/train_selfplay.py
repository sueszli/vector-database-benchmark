import os
from datetime import datetime
import numpy as np
import torch
import gym
import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import DEFAULT_LOGGERS
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Pred
from gym_predprey.envs.PredPrey1v1 import PredPrey1v1Prey
OBS = 'full'
ACT = 'vel'
ENV = 'PredPrey-Pred-v0'
WORKERS = 1
ALGO = 'PPO'
SEED_VALUE = 3
TRAINING_ITERATION = 1000
PRED_TRAINING_EPOCHS = 5
PREY_TRAINING_EPOCHS = 5
checkpoint_dir = None

def make_deterministic(seed):
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True

class InitAgent:

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        self.env = env

    def compute_action(self, _):
        if False:
            print('Hello World!')
        return self.env.action_space.sample()

class SelfPlayPolicies:

    def __init__(self, initialization_policy, num_policies=2, keys=None):
        if False:
            print('Hello World!')
        self.num_policies = num_policies
        if keys is None:
            self.keys = [i for i in range(num_policies)]
        else:
            if num_policies != len(keys):
                print('Number of policies is not equal to number of keys provideed to map the policy to the dictionary')
                raise ValueError
            self.keys = keys
        self.policies = {self.keys[i]: [{'policy': initialization_policy[self.keys[i]], 'path': None}] for i in range(num_policies)}

    def sample(self, num_sampled_policies=1):
        if False:
            for i in range(10):
                print('nop')
        policies = {}
        for i in range(self.num_policies):
            key = self.keys[i]
            policies[key] = np.random.choice(self.policies[key], num_sampled_policies)
        return policies

    def store(self, policies, path):
        if False:
            while True:
                i = 10
        for i in range(self.num_policies):
            key = self.keys[i]
            self.policies[key].append({'policy': policies[key], 'path': path[key]})

    def get_num_policies(self):
        if False:
            print('Hello World!')
        return len(self.policies[self.keys[0]])

def selfplay_train_func(config, reporter):
    if False:
        return 10
    selfplay_policies = config['env_config']['something']
    sampled_policies = selfplay_policies.sample()
    sampled_pred = sampled_policies['pred'][0]['policy']
    sampled_prey = sampled_policies['prey'][0]['policy']
    agent1 = PPOTrainer(env='CartPole-v0', config=config)
    for _ in range(10):
        result = agent1.train()
        result['phase'] = 1
        reporter(**result)
        phase1_time = result['timesteps_total']
    state = agent1.save()
    agent1.stop()
    config['lr'] = 0.0001
    agent2 = PPOTrainer(env='CartPole-v0', config=config)
    agent2.restore(state)
    for _ in range(10):
        result = agent2.train()
        result['phase'] = 2
        result['timesteps_total'] += phase1_time
        reporter(**result)
    agent2.stop()

def selfplay_train_func_test(config, reporter):
    if False:
        print('Hello World!')
    sampled_policies = selfplay_policies.sample()
    sampled_pred = sampled_policies['pred'][0]['policy']
    sampled_prey = sampled_policies['prey'][0]['policy']
    print('-------------- Train Predator --------------------')
    config['env_config'] = {'prey_policy': sampled_prey}
    print(pretty_print(config))
    register_env('Pred', lambda _: PredPrey1v1Pred(prey_policy=sampled_prey))
    pred_agent = PPOTrainer(env='Pred', config=config)
    if selfplay_policies.get_num_policies() > 1:
        pred_agent.restore(sampled_policies['pred'][0]['path'])
    for pred_epoch in range(PRED_TRAINING_EPOCHS):
        result = pred_agent.train()
        result['phase'] = 'Predator'
        reporter(**result)
        pred_time = result['timesteps_total']
    state = pred_agent.save()
    pred_save_checkpoint = pred_agent.save_checkpoint(checkpoint_dir + '-pred')
    pred_agent.stop()
    print('-------------- Train Prey --------------------')
    config['env_config'] = {'pred_policy': sampled_pred}
    register_env('Prey', lambda _: PredPrey1v1Prey(pred_policy=sampled_pred))
    prey_agent = PPOTrainer(PPOTrainer='Prey', config=config)
    if selfplay_policies.get_num_policies() > 1:
        prey_agent.restore(sampled_policies['prey'][0]['path'])
    for prey_epoch in range(PREY_TRAINING_EPOCHS):
        result = prey_agent.train()
        result['phase'] = 'Prey'
        result['timesteps_total'] += pred_time
        reporter(**result)
    state = prey_agent.save()
    prey_save_checkpoint = prey_agent.save_checkpoint(checkpoint_dir + '-prey')
    prey_agent.stop()
    selfplay_policies.store({'pred': pred_agent, 'prey': prey_agent}, path={'pred': pred_save_checkpoint, 'prey': prey_save_checkpoint})
    print('------------------------------------------------------')
if __name__ == '__main__':
    if torch.cuda.is_available():
        print('## CUDA available')
        print(f'Current device: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('## CUDA not available')
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__)) + '/selfplay-results/save-' + ENV + '-' + ALGO + '-' + OBS + '-' + ACT + '-' + datetime.now().strftime('%m.%d.%Y_%H.%M.%S')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir + '/')
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    pred_agent = InitAgent(env=PredPrey1v1Pred())
    prey_agent = InitAgent(env=PredPrey1v1Prey())
    selfplay_policies = SelfPlayPolicies(initialization_policy={'pred': pred_agent, 'prey': prey_agent}, num_policies=2, keys=['pred', 'prey'])
    ppo_default_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config = ppo_default_config
    config = {'num_workers': 0 + WORKERS, 'num_gpus': 1 if torch.cuda.is_available() else 0, 'batch_mode': 'complete_episodes', 'seed': SEED_VALUE, 'framework': 'torch', 'env_config': {'something': selfplay_policies}}
    pretty_print(config)
    stop = {'training_iteration': TRAINING_ITERATION}
    config = {**ppo_config, **config}
    results = tune.run(selfplay_train_func, stop=stop, config=config, log_to_file=['logs_out.txt', 'logs_err.txt'], checkpoint_freq=5, checkpoint_at_end=True, local_dir=checkpoint_dir, resources_per_trial=PPOTrainer.default_resource_request(config), loggers=DEFAULT_LOGGERS)
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean', mode='max'), metric='episode_reward_mean')
    with open(checkpoint_dir + '/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])
    ray.shutdown()