import argparse
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ignite.engine import Engine, Events
try:
    import gymnasium as gym
except ImportError:
    raise ModuleNotFoundError('Please install opengym: pip install gymnasium')
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        '\n        forward of both actor and critic\n        '
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return (action_prob, state_values)

def select_action(policy, observation):
    if False:
        while True:
            i = 10
    observation = torch.from_numpy(observation).float()
    (probs, observation_value) = policy(observation)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(SavedAction(m.log_prob(action), observation_value))
    return action.item()

def finish_episode(policy, optimizer, gamma):
    if False:
        i = 10
        return i + 15
    '\n    Training code. Calculates actor and critic loss and performs backprop.\n    '
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []
    value_losses = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for ((log_prob, value), R) in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]
EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

def main(env, args):
    if False:
        while True:
            i = 10
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=0.03)
    timesteps = range(10000)

    def run_single_timestep(engine, timestep):
        if False:
            while True:
                i = 10
        observation = engine.state.observation
        action = select_action(policy, observation)
        (engine.state.observation, reward, done, _, _) = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        engine.state.ep_reward += reward
        if done:
            engine.terminate_epoch()
            engine.state.timestep = timestep
    trainer = Engine(run_single_timestep)
    trainer.state.running_reward = 10

    @trainer.on(EPISODE_STARTED)
    def reset_environment_state():
        if False:
            print('Hello World!')
        torch.manual_seed(args.seed + trainer.state.epoch)
        (trainer.state.observation, _) = env.reset(seed=args.seed + trainer.state.epoch)
        trainer.state.ep_reward = 0

    @trainer.on(EPISODE_COMPLETED)
    def update_model():
        if False:
            while True:
                i = 10
        t = trainer.state.timestep
        trainer.state.running_reward = 0.05 * trainer.state.ep_reward + (1 - 0.05) * trainer.state.running_reward
        finish_episode(policy, optimizer, args.gamma)

    @trainer.on(EPISODE_COMPLETED(every=args.log_interval))
    def log_episode():
        if False:
            for i in range(10):
                print('nop')
        i_episode = trainer.state.epoch
        print(f'Episode {i_episode}\tLast reward: {trainer.state.ep_reward:.2f}\tAverage reward: {trainer.state.running_reward:.2f}')

    @trainer.on(EPISODE_COMPLETED)
    def should_finish_training():
        if False:
            while True:
                i = 10
        running_reward = trainer.state.running_reward
        if running_reward > env.spec.reward_threshold:
            print(f'Solved! Running reward is now {running_reward} and the last episode runs to {trainer.state.timestep} time steps!')
            trainer.should_terminate = True
    trainer.run(timesteps, max_epochs=args.max_episodes)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ignite actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--max-episodes', type=int, default=1000000, metavar='N', help='Number of episodes for the training (default: 1000000)')
    args = parser.parse_args()
    env = gym.make('CartPole-v1')
    main(env, args)