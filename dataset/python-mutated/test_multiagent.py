"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `multiagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
import shared_constants
OWN_OBS_VEC_SIZE = None
ACTION_VEC_SIZE = None

class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            for i in range(10):
                print('nop')
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.action_model = FullyConnectedNetwork(Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE,)), action_space, num_outputs, model_config, name + '_action')
        self.value_model = FullyConnectedNetwork(obs_space, action_space, 1, model_config, name + '_vf')
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        if False:
            for i in range(10):
                print('nop')
        self._model_in = [input_dict['obs_flat'], state, seq_lens]
        return self.action_model({'obs': input_dict['obs']['own_obs']}, state, seq_lens)

    def value_function(self):
        if False:
            for i in range(10):
                print('nop')
        (value_out, _) = self.value_model({'obs': self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])

class FillInActions(DefaultCallbacks):

    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        if False:
            i = 10
            return i + 15
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Box(-1, 1, (ACTION_VEC_SIZE,), np.float32))
        (_, opponent_batch) = original_batches[other_id]
        opponent_actions = np.array([action_encoder.transform(np.clip(a, -1, 1)) for a in opponent_batch[SampleBatch.ACTIONS]])
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

def central_critic_observer(agent_obs, **kw):
    if False:
        while True:
            i = 10
    new_obs = {0: {'own_obs': agent_obs[0], 'opponent_obs': agent_obs[1], 'opponent_action': np.zeros(ACTION_VEC_SIZE)}, 1: {'own_obs': agent_obs[1], 'opponent_obs': agent_obs[0], 'opponent_action': np.zeros(ACTION_VEC_SIZE)}}
    return new_obs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp', type=str, help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()
    NUM_DRONES = int(ARGS.exp.split('-')[2])
    OBS = ObservationType.KIN if ARGS.exp.split('-')[4] == 'kin' else ObservationType.RGB
    if ARGS.exp.split('-')[5] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split('-')[5] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split('-')[5] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split('-')[5] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split('-')[5] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split('-')[5] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split('-')[5] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID
    if OBS == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 12
    elif OBS == ObservationType.RGB:
        print('[ERROR] ObservationType.RGB for multi-agent systems not yet implemented')
        exit()
    else:
        print('[ERROR] unknown ObservationType')
        exit()
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ACT == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print('[ERROR] unknown ActionType')
        exit()
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model('cc_model', CustomTorchCentralizedCriticModel)
    temp_env_name = 'this-aviary-v0'
    if ARGS.exp.split('-')[1] == 'flock':
        register_env(temp_env_name, lambda _: FlockAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT))
    elif ARGS.exp.split('-')[1] == 'leaderfollower':
        register_env(temp_env_name, lambda _: LeaderFollowerAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT))
    elif ARGS.exp.split('-')[1] == 'meetup':
        register_env(temp_env_name, lambda _: MeetupAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT))
    else:
        print('[ERROR] environment not yet implemented')
        exit()
    if ARGS.exp.split('-')[1] == 'flock':
        temp_env = FlockAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT)
    elif ARGS.exp.split('-')[1] == 'leaderfollower':
        temp_env = LeaderFollowerAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT)
    elif ARGS.exp.split('-')[1] == 'meetup':
        temp_env = MeetupAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT)
    else:
        print('[ERROR] environment not yet implemented')
        exit()
    observer_space = Dict({'own_obs': temp_env.observation_space[0], 'opponent_obs': temp_env.observation_space[0], 'opponent_action': temp_env.action_space[0]})
    action_space = temp_env.action_space[0]
    config = ppo.DEFAULT_CONFIG.copy()
    config = {'env': temp_env_name, 'num_workers': 0, 'num_gpus': int(os.environ.get('RLLIB_NUM_GPUS', '0')), 'batch_mode': 'complete_episodes', 'callbacks': FillInActions, 'framework': 'torch'}
    config['model'] = {'custom_model': 'cc_model'}
    config['multiagent'] = {'policies': {'pol0': (None, observer_space, action_space, {'agent_id': 0}), 'pol1': (None, observer_space, action_space, {'agent_id': 1})}, 'policy_mapping_fn': lambda x: 'pol0' if x == 0 else 'pol1', 'observation_fn': central_critic_observer}
    agent = ppo.PPOTrainer(config=config)
    with open(ARGS.exp + '/checkpoint.txt', 'r+') as f:
        checkpoint = f.read()
    agent.restore(checkpoint)
    policy0 = agent.get_policy('pol0')
    print('action model 0', policy0.model.action_model)
    print('value model 0', policy0.model.value_model)
    policy1 = agent.get_policy('pol1')
    print('action model 1', policy1.model.action_model)
    print('value model 1', policy1.model.value_model)
    if ARGS.exp.split('-')[1] == 'flock':
        test_env = FlockAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT, gui=True, record=False)
    elif ARGS.exp.split('-')[1] == 'leaderfollower':
        test_env = LeaderFollowerAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT, gui=True, record=False)
    elif ARGS.exp.split('-')[1] == 'meetup':
        test_env = MeetupAviary(num_drones=NUM_DRONES, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT, gui=True, record=False)
    else:
        print('[ERROR] environment not yet implemented')
        exit()
    obs = test_env.reset()
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS), num_drones=NUM_DRONES)
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        action = {i: np.array([0]) for i in range(NUM_DRONES)}
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
    elif ACT == ActionType.PID:
        action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
    else:
        print('[ERROR] unknown ActionType')
        exit()
    start = time.time()
    for i in range(6 * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):
        temp = {}
        temp[0] = policy0.compute_single_action(np.hstack([action[1], obs[1], obs[0]]))
        temp[1] = policy1.compute_single_action(np.hstack([action[0], obs[0], obs[1]]))
        action = {0: temp[0][0]}
        for i in range(1, NUM_DRONES):
            action[i] = temp[1][0]
        (obs, reward, done, info) = test_env.step(action)
        test_env.render()
        if OBS == ObservationType.KIN:
            for j in range(NUM_DRONES):
                logger.log(drone=j, timestamp=i / test_env.SIM_FREQ, state=np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(action[j], 4)]), control=np.zeros(12))
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
    test_env.close()
    logger.save_as_csv('ma')
    logger.plot()
    ray.shutdown()