"""This example script shows how to load a connector enabled policy,
and use it in a serving/inference setting.
"""
import gymnasium as gym
import os
import tempfile
from ray.rllib.examples.connectors.prepare_checkpoint import create_appo_cartpole_checkpoint
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.policy import local_policy_inference

def run(checkpoint_path, policy_id):
    if False:
        print('Hello World!')
    policy = Policy.from_checkpoint(checkpoint=checkpoint_path, policy_ids=[policy_id])
    env = gym.make('CartPole-v1')
    (obs, info) = env.reset()
    terminated = truncated = False
    step = 0
    while not terminated and (not truncated):
        step += 1
        policy_outputs = local_policy_inference(policy, 'env_1', 'agent_1', obs, explore=False)
        assert len(policy_outputs) == 1
        (action, _, _) = policy_outputs[0]
        print(f'step {step}', obs, action)
        (obs, _, terminated, truncated, _) = env.step(action)
if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_id = 'default_policy'
        create_appo_cartpole_checkpoint(tmpdir)
        policy_checkpoint_path = os.path.join(tmpdir, 'policies', policy_id)
        run(policy_checkpoint_path, policy_id)