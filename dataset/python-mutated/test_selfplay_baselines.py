import os
import argparse
from datetime import datetime
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPredEnv
from gym_predprey.envs.SelfPlayPredPrey1v1 import SelfPlayPreyEnv
from time import sleep
OBS = 'full'
ACT = 'vel'
ENV = 'PredPrey-Pred-v0'
WORKERS = 1
ALGO = 'PPO'
SEED_VALUE = 3
EVAL_EPISODES = 5
LOG_DIR = None
NUM_TESTING_EPISODES = 5
RENDER_MODE = True

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

def rollout(env, policy):
    if False:
        i = 10
        return i + 15
    ' play one agent vs the other in modified gym-style loop. '
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        (action, _states) = policy.predict(obs)
        (obs, reward, done, info) = env.step(action)
        total_reward += reward
        if RENDER_MODE:
            env.render()
            sleep(0.005)
    return (total_reward, info['win'])

class PPOMod(PPO):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(PPOMod, self).__init__(*args, **kwargs)

    def load(model_path, env):
        if False:
            for i in range(10):
                print('nop')
        custom_objects = {'lr_schedule': lambda x: 0.003, 'clip_range': lambda x: 0.02}
        return PPO.load(model_path, env, custom_objects=custom_objects)

def test(log_dir):
    if False:
        return 10
    logger.configure(folder=log_dir)
    pred_env = SelfPlayPredEnv(log_dir=log_dir, algorithm_class=PPOMod)
    pred_env.set_target_opponent_policy_filename(os.path.join(log_dir, 'prey', 'final_model'))
    pred_env.seed(SEED_VALUE)
    pred_model = PPOMod.load(os.path.join(log_dir, 'pred', 'final_model'), pred_env)
    rewards = []
    winner = []
    for i in range(NUM_TESTING_EPISODES):
        (r, w) = rollout(pred_env, pred_model)
        w = 'pred' if w > 0 else 'prey'
        rewards.append(r)
        winner.append(w)
        print(f'Winner: {w} -> reward: {r}')
    print(list(zip(winner, rewards)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp', type=str, help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()
    LOG_DIR = ARGS.exp
    test(LOG_DIR)