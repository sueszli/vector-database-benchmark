"""This example script shows how one can use Ray Serve to serve an already
trained RLlib Policy (and its model) to serve action computations.

For a complete tutorial, also see:
https://docs.ray.io/en/master/serve/tutorials/rllib.html
"""
import argparse
import gymnasium as gym
import requests
from starlette.requests import Request
import ray
import ray.rllib.algorithms.algorithm as Algorithm
import ray.rllib.algorithms.algorithm_config as AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.atari_wrappers import FrameStack, WarpFrame
from ray import serve
parser = argparse.ArgumentParser()
parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
parser.add_argument('--train-iters', type=int, default=1)
parser.add_argument('--no-render', action='store_true')
args = parser.parse_args()

class ServeRLlibPolicy:
    """Callable class used by Ray Serve to handle async requests.

    All the necessary serving logic is implemented in here:
    - Creation and restoring of the (already trained) RLlib Algorithm.
    - Calls to algo.compute_action upon receiving an action request
      (with a current observation).
    """

    def __init__(self, checkpoint_path):
        if False:
            return 10
        self.algo = Algorithm.from_checkpoint(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input['observation']
        action = self.algo.compute_single_action(obs)
        return {'action': int(action)}

def train_rllib_policy(config: AlgorithmConfig):
    if False:
        for i in range(10):
            print('nop')
    'Trains a DQN on ALE/MsPacman-v5 for n iterations.\n\n    Saves the trained Algorithm to disk and returns the checkpoint path.\n\n    Args:\n        config: The algo config object for the Algorithm.\n\n    Returns:\n        str: The saved checkpoint to restore DQN from.\n    '
    algo = config.build()
    for _ in range(args.train_iters):
        print(algo.train())
    checkpoint_result = algo.save()
    algo.stop()
    return checkpoint_result
if __name__ == '__main__':
    config = DQNConfig().environment('ALE/MsPacman-v5').framework(args.framework)
    checkpoint_result = train_rllib_policy(config)
    ray.init(num_cpus=8)
    client = serve.start()
    client.create_backend('backend', ServeRLlibPolicy, config, checkpoint_result)
    client.create_endpoint('endpoint', backend='backend', route='/mspacman-rllib-policy')
    env = FrameStack(WarpFrame(gym.make('ALE/MsPacman-v5'), 84), 4)
    (obs, info) = env.reset()
    while True:
        print('-> Requesting action for obs ...')
        resp = requests.get('http://localhost:8000/mspacman-rllib-policy', json={'observation': obs.tolist()})
        response = resp.json()
        print('<- Received response {}'.format(response))
        action = response['action']
        (obs, reward, done, _, _) = env.step(action)
        if done:
            (obs, info) = env.reset()
        if not args.no_render:
            env.render()