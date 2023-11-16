"""Deep Reinforcement Learning: Deep Q-network (DQN)

The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.

To run the template, just run:
`python reinforce_learn_Qnet.py`

After ~1500 steps, you will see the total_reward hitting the max score of 475+.
Open up TensorBoard to see the metrics:

`tensorboard --logdir default`

References
----------

[1] https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py

"""
import argparse
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule, Trainer, cli_lightning_logo, seed_everything
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

class DQN(nn.Module):
    """Simple MLP network.

    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )

    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int=128):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            obs_size: observation/state size of the environment\n            n_actions: number of discrete actions available in the environment\n            hidden_size: size of hidden layers\n        '
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.net(x.float())
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>

    """

    def __init__(self, capacity: int) -> None:
        if False:
            return 10
        '\n        Args:\n            capacity: size of the buffer\n        '
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        if False:
            return 10
        'Add experience to the buffer.\n\n        Args:\n            experience: tuple (state, action, reward, done, new_state)\n\n        '
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        if False:
            print('Hello World!')
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        (states, actions, rewards, dones, next_states) = zip(*(self.buffer[idx] for idx in indices))
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool), np.array(next_states))

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    >>> RLDataset(ReplayBuffer(5))  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.RLDataset object at ...>

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int=200) -> None:
        if False:
            return 10
        '\n        Args:\n            buffer: replay buffer\n            sample_size: number of experiences to sample at a time\n        '
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        if False:
            while True:
                i = 10
        (states, actions, rewards, dones, new_states) = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield (states[i], actions[i], rewards[i], dones[i], new_states[i])

class Agent:
    """Base Agent class handling the interaction with the environment.

    >>> env = gym.make("CartPole-v1")
    >>> buffer = ReplayBuffer(10)
    >>> Agent(env, buffer)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.Agent object at ...>

    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        if False:
            return 10
        '\n        Args:\n            env: training environment\n            replay_buffer: replay buffer storing experiences\n        '
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        'Resets the environment and updates the state.'
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        if False:
            print('Hello World!')
        'Using the given network, decide what action to carry out using an epsilon-greedy policy.\n\n        Args:\n            net: DQN network\n            epsilon: value to determine likelihood of taking a random action\n            device: current device\n\n        Returns:\n            action\n\n        '
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])
            if device not in ['cpu']:
                state = state.cuda(device)
            q_values = net(state)
            (_, action) = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float=0.0, device: str='cpu') -> Tuple[float, bool]:
        if False:
            i = 10
            return i + 15
        'Carries out a single interaction step between the agent and the environment.\n\n        Args:\n            net: DQN network\n            epsilon: value to determine likelihood of taking a random action\n            device: current device\n\n        Returns:\n            reward, done\n\n        '
        action = self.get_action(net, epsilon, device)
        (new_state, reward, done, _) = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        return (reward, done)

class DQNLightning(LightningModule):
    """Basic DQN Model.

    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )

    """

    def __init__(self, env: str, replay_size: int=200, warm_start_steps: int=200, gamma: float=0.99, eps_start: float=1.0, eps_end: float=0.01, eps_last_frame: int=200, sync_rate: int=10, lr: float=0.01, episode_length: int=50, batch_size: int=4, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)
        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.warm_start_steps)

    def populate(self, steps: int=1000) -> None:
        if False:
            return 10
        'Carries out several random steps through the environment to initially fill up the replay buffer with\n        experiences.\n\n        Args:\n            steps: number of random steps to populate the buffer with\n\n        '
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        'Passes in a state `x` through the network and gets the `q_values` of each action as an output.\n\n        Args:\n            x: environment state\n\n        Returns:\n            q values\n\n        '
        return self.net(x)

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Calculates the mse loss using a mini batch from the replay buffer.\n\n        Args:\n            batch: current mini batch of replay data\n\n        Returns:\n            loss\n\n        '
        (states, actions, rewards, dones, next_states) = batch
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        if False:
            for i in range(10):
                print('nop')
        'Carries out a single step through the environment to update the replay buffer. Then calculates loss based on\n        the minibatch received.\n\n        Args:\n            batch: current mini batch of replay data\n            nb_batch: batch number\n\n        Returns:\n            Training loss and log metrics\n\n        '
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start - (self.global_step + 1) / self.eps_last_frame)
        (reward, done) = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        loss = self.dqn_mse_loss(batch)
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        log = {'total_reward': torch.tensor(self.total_reward).to(device), 'reward': torch.tensor(reward).to(device), 'steps': torch.tensor(self.global_step).to(device)}
        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})

    def configure_optimizers(self) -> List[Optimizer]:
        if False:
            print('Hello World!')
        'Initialize Adam optimizer.'
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the Replay Buffer dataset used for retrieving experiences.'
        dataset = RLDataset(self.buffer, self.episode_length)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, sampler=None)

    def train_dataloader(self) -> DataLoader:
        if False:
            for i in range(10):
                print('nop')
        'Get train loader.'
        return self.__dataloader()

    def get_device(self, batch) -> str:
        if False:
            print('Hello World!')
        'Retrieve device currently being used by minibatch.'
        return batch[0].device.index if self.on_gpu else 'cpu'

def main(args) -> None:
    if False:
        return 10
    model = DQNLightning(**vars(args))
    trainer = Trainer(accelerator='cpu', devices=1, val_check_interval=100)
    trainer.fit(model)
if __name__ == '__main__':
    cli_lightning_logo()
    seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='gym environment tag')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--sync_rate', type=int, default=10, help='how many frames do we update the target network')
    parser.add_argument('--replay_size', type=int, default=1000, help='capacity of the replay buffer')
    parser.add_argument('--warm_start_steps', type=int, default=1000, help='how many samples do we use to fill our buffer at the start of training')
    parser.add_argument('--eps_last_frame', type=int, default=1000, help='what frame should epsilon stop decaying')
    parser.add_argument('--eps_start', type=float, default=1.0, help='starting value of epsilon')
    parser.add_argument('--eps_end', type=float, default=0.01, help='final value of epsilon')
    parser.add_argument('--episode_length', type=int, default=200, help='max length of an episode')
    args = parser.parse_args()
    main(args)