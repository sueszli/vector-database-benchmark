"""Example of using rollout worker classes directly to implement training.

Instead of using the built-in Algorithm classes provided by RLlib, here we define
a custom Policy class and manually coordinate distributed sample
collection and policy optimization.
"""
import argparse
import gymnasium as gym
import numpy as np
import ray
from ray import train, tune
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.tune.execution.placement_groups import PlacementGroupFactory
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--num-iters', type=int, default=20)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--num-cpus', type=int, default=0)

class CustomPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to extend TF/TorchPolicy instead
    for a real policy.
    """

    def __init__(self, observation_space, action_space, config):
        if False:
            while True:
                i = 10
        super().__init__(observation_space, action_space, config)
        self.config['framework'] = None
        self.w = 1.0

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        if False:
            return 10
        return (np.array([self.action_space.sample() for _ in obs_batch]), [], {})

    def learn_on_batch(self, samples):
        if False:
            while True:
                i = 10
        return {}

    def update_some_value(self, w):
        if False:
            print('Hello World!')
        self.w = w

    def get_weights(self):
        if False:
            for i in range(10):
                print('nop')
        return {'w': self.w}

    def set_weights(self, weights):
        if False:
            while True:
                i = 10
        self.w = weights['w']

def training_workflow(config):
    if False:
        for i in range(10):
            print('nop')
    env = gym.make('CartPole-v1')
    policy = CustomPolicy(env.observation_space, env.action_space, {})
    workers = [ray.remote()(RolloutWorker).remote(env_creator=lambda c: gym.make('CartPole-v1'), policy=CustomPolicy) for _ in range(config['num_workers'])]
    for _ in range(config['num_iters']):
        weights = ray.put({DEFAULT_POLICY_ID: policy.get_weights()})
        for w in workers:
            w.set_weights.remote(weights)
        T1 = concat_samples(ray.get([w.sample.remote() for w in workers]))
        new_value = policy.w * 2.0
        for w in workers:
            w.for_policy.remote(lambda p: p.update_some_value(new_value))
        T2 = concat_samples(ray.get([w.sample.remote() for w in workers]))
        policy.learn_on_batch(T1)
        policy.update_some_value(sum(T2['rewards']))
        train.report(collect_metrics(remote_workers=workers))
if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None)
    tune.Tuner(tune.with_resources(training_workflow, resources=PlacementGroupFactory([{'CPU': 1, 'GPU': 1 if args.gpu else 0}] + [{'CPU': 1}] * args.num_workers)), param_space={'num_workers': args.num_workers, 'num_iters': args.num_iters}, run_config=train.RunConfig(verbose=1))