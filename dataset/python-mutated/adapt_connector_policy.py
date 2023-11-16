"""This example script shows how to load a connector enabled policy,
and adapt/use it with a different version of the environment.
"""
import gymnasium as gym
import numpy as np
import os
import tempfile
from typing import Dict
from ray.rllib.connectors.connector import ConnectorContext
from ray.rllib.connectors.action.lambdas import register_lambda_action_connector
from ray.rllib.connectors.agent.lambdas import register_lambda_agent_connector
from ray.rllib.examples.connectors.prepare_checkpoint import create_appo_cartpole_checkpoint
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.policy import local_policy_inference
from ray.rllib.utils.typing import PolicyOutputType, StateBatches, TensorStructType

class MyCartPole(gym.Env):
    """A mock CartPole environment.

    Gives 2 additional observation states and takes 2 discrete actions.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._env = gym.make('CartPole-v1')
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,))
        self.action_space = gym.spaces.MultiDiscrete(nvec=[2, 2])

    def step(self, actions):
        if False:
            print('Hello World!')
        action = actions[0]
        (obs, reward, done, truncated, info) = self._env.step(action)
        obs = np.hstack((obs, [8.0, 6.0]))
        return (obs, reward, done, truncated, info)

    def reset(self, *, seed=None, options=None):
        if False:
            print('Hello World!')
        (obs, info) = self._env.reset()
        return (np.hstack((obs, [8.0, 6.0])), info)

def v2_to_v1_obs(data: Dict[str, TensorStructType]) -> Dict[str, TensorStructType]:
    if False:
        for i in range(10):
            print('nop')
    data[SampleBatch.NEXT_OBS] = data[SampleBatch.NEXT_OBS][:-2]
    return data
V2ToV1ObsAgentConnector = register_lambda_agent_connector('V2ToV1ObsAgentConnector', v2_to_v1_obs)

def v1_to_v2_action(actions: TensorStructType, states: StateBatches, fetches: Dict) -> PolicyOutputType:
    if False:
        i = 10
        return i + 15
    return (np.hstack((actions, [0])), states, fetches)
V1ToV2ActionConnector = register_lambda_action_connector('V1ToV2ActionConnector', v1_to_v2_action)

def run(checkpoint_path, policy_id):
    if False:
        while True:
            i = 10
    policy = Policy.from_checkpoint(checkpoint=checkpoint_path, policy_ids=[policy_id])
    ctx: ConnectorContext = ConnectorContext.from_policy(policy)
    policy.agent_connectors.remove('FlattenDataAgentConnector')
    policy.agent_connectors.prepend(V2ToV1ObsAgentConnector(ctx))
    policy.action_connectors.append(V1ToV2ActionConnector(ctx))
    env = MyCartPole()
    (obs, info) = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        policy_outputs = local_policy_inference(policy, 'env_1', 'agent_1', obs)
        assert len(policy_outputs) == 1
        (actions, _, _) = policy_outputs[0]
        print(f'step {step}', obs, actions)
        (obs, _, done, _, _) = env.step(actions)
if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_id = 'default_policy'
        create_appo_cartpole_checkpoint(tmpdir)
        policy_checkpoint_path = os.path.join(tmpdir, 'policies', policy_id)
        run(policy_checkpoint_path, policy_id)