"""PyTorch Lightning implementation of Proximal Policy Optimization (PPO)

<https://arxiv.org/abs/1707.06347>
Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

The example implements PPO compatible to work with any continuous or discrete action-space environments via OpenAI Gym.

To run the template, just run:
`python reinforce_learn_ppo.py`

References
----------
[1] https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
[2] https://github.com/openai/spinningup
[3] https://github.com/sid-sundrani/ppo_lightning

"""
import argparse
from typing import Callable, Iterator, List, Tuple
import gym
import torch
from lightning.pytorch import LightningModule, Trainer, cli_lightning_logo, seed_everything
from torch import nn
from torch.distributions import Categorical, Normal
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

def create_mlp(input_shape: Tuple[int], n_actions: int, hidden_size: int=128):
    if False:
        i = 10
        return i + 15
    'Simple Multi-Layer Perceptron network.'
    return nn.Sequential(nn.Linear(input_shape[0], hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

class ActorCategorical(nn.Module):
    """Policy network, for discrete action spaces, which returns a distribution and an action given an observation."""

    def __init__(self, actor_net):
        if False:
            while True:
                i = 10
        '\n        Args:\n            input_shape: observation shape of the environment\n            n_actions: number of discrete actions available in the environment\n        '
        super().__init__()
        self.actor_net = actor_net

    def forward(self, states):
        if False:
            for i in range(10):
                print('nop')
        logits = self.actor_net(states)
        pi = Categorical(logits=logits)
        actions = pi.sample()
        return (pi, actions)

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        if False:
            print('Hello World!')
        'Takes in a distribution and actions and returns log prob of actions under the distribution.\n\n        Args:\n            pi: torch distribution\n            actions: actions taken by distribution\n\n        Returns:\n            log probability of the action under pi\n\n        '
        return pi.log_prob(actions)

class ActorContinuous(nn.Module):
    """Policy network, for continuous action spaces, which returns a distribution and an action given an
    observation."""

    def __init__(self, actor_net, act_dim):
        if False:
            return 10
        '\n        Args:\n            input_shape: observation shape of the environment\n            n_actions: number of discrete actions available in the environment\n        '
        super().__init__()
        self.actor_net = actor_net
        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float)
        self.log_std = nn.Parameter(log_std)

    def forward(self, states):
        if False:
            print('Hello World!')
        mu = self.actor_net(states)
        std = torch.exp(self.log_std)
        pi = Normal(loc=mu, scale=std)
        actions = pi.sample()
        return (pi, actions)

    def get_log_prob(self, pi: Normal, actions: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        'Takes in a distribution and actions and returns log prob of actions under the distribution.\n\n        Args:\n            pi: torch distribution\n            actions: actions taken by distribution\n\n        Returns:\n            log probability of the action under pi\n\n        '
        return pi.log_prob(actions).sum(axis=-1)

class ExperienceSourceDataset(IterableDataset):
    """Implementation from PyTorch Lightning Bolts: https://github.com/Lightning-AI/lightning-
    bolts/blob/master/pl_bolts/datamodules/experience_source.py.

    Basic experience source dataset. Takes a generate_batch function that returns an iterator. The logic for the
    experience source and how the batch is generated is defined the Lightning model itself

    """

    def __init__(self, generate_batch: Callable):
        if False:
            while True:
                i = 10
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        if False:
            return 10
        return self.generate_batch()

class PPOLightning(LightningModule):
    """PyTorch Lightning implementation of PPO.

    Example:
        model = PPOLightning("CartPole-v0")
    Train:
        trainer = Trainer()
        trainer.fit(model)

    """

    def __init__(self, env: str, gamma: float=0.99, lam: float=0.95, lr_actor: float=0.0003, lr_critic: float=0.001, max_episode_len: float=200, batch_size: int=512, steps_per_epoch: int=2048, nb_optim_iters: int=4, clip_ratio: float=0.2, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            env: gym environment tag\n            gamma: discount factor\n            lam: advantage discount factor (lambda in the paper)\n            lr_actor: learning rate of actor network\n            lr_critic: learning rate of critic network\n            max_episode_len: maximum number interactions (actions) in an episode\n            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch\n            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch\n            nb_optim_iters: how many steps of gradient descent to perform on each batch\n            clip_ratio: hyperparameter for clipping in the policy objective\n        '
        super().__init__()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.env = gym.make(env)
        self.critic = create_mlp(self.env.observation_space.shape, 1)
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            act_dim = self.env.action_space.shape[0]
            actor_mlp = create_mlp(self.env.observation_space.shape, act_dim)
            self.actor = ActorContinuous(actor_mlp, act_dim)
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            actor_mlp = create_mlp(self.env.observation_space.shape, self.env.action_space.n)
            self.actor = ActorCategorical(actor_mlp)
        else:
            raise NotImplementedError(f'Env action space should be of type Box (continuous) or Discrete (categorical). Got type: {type(self.env.action_space)}')
        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []
        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []
        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0
        self.state = torch.FloatTensor(self.env.reset())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        'Passes in a state x through the network and returns the policy and a sampled action.\n\n        Args:\n            x: environment state\n\n        Returns:\n            Tuple of policy and action\n\n        '
        (pi, action) = self.actor(x)
        value = self.critic(x)
        return (pi, action, value)

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        if False:
            while True:
                i = 10
        'Calculate the discounted rewards of all rewards in list.\n\n        Args:\n            rewards: list of rewards/advantages\n\n        Returns:\n            list of discounted rewards/advantages\n\n        '
        assert isinstance(rewards[0], float)
        cumul_reward = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r = sum_r * discount + r
            cumul_reward.append(sum_r)
        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        if False:
            print('Hello World!')
        'Calculate the advantage given rewards, state values, and the last value of episode.\n\n        Args:\n            rewards: list of episode rewards\n            values: list of state values from critic\n            last_value: value of last state of episode\n\n        Returns:\n            list of advantages\n\n        '
        rews = rewards + [last_value]
        vals = values + [last_value]
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        return self.discount_rewards(delta, self.gamma * self.lam)

    def generate_trajectory_samples(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Contains the logic for generating trajectory data to train policy and value network\n        Yield:\n           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage\n        '
        for step in range(self.steps_per_epoch):
            self.state = self.state.to(device=self.device)
            with torch.no_grad():
                (pi, action, value) = self(self.state)
                log_prob = self.actor.get_log_prob(pi, action)
            (next_state, reward, done, _) = self.env.step(action.cpu().numpy())
            self.episode_step += 1
            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)
            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())
            self.state = torch.FloatTensor(next_state)
            epoch_end = step == self.steps_per_epoch - 1
            terminal = len(self.ep_rewards) == self.max_episode_len
            if epoch_end or done or terminal:
                if (terminal or epoch_end) and (not done):
                    self.state = self.state.to(device=self.device)
                    with torch.no_grad():
                        (_, _, value) = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                self.epoch_rewards.append(sum(self.ep_rewards))
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())
            if epoch_end:
                train_data = zip(self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv)
                for (state, action, logp_old, qval, adv) in train_data:
                    yield (state, action, logp_old, qval, adv)
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]
                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)
                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes
                self.epoch_rewards.clear()

    def actor_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        (pi, _) = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        return -torch.min(ratio * adv, clip_adv).mean()

    def critic_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        if False:
            return 10
        value = self.critic(state)
        return (qval - value).pow(2).mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        if False:
            print('Hello World!')
        'Carries out a single update to actor and critic network from a batch of replay buffer.\n\n        Args:\n            batch: batch of replay buffer/trajectory data\n\n        '
        (state, action, old_logp, qval, adv) = batch
        adv = (adv - adv.mean()) / adv.std()
        self.log('avg_ep_len', self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        self.log('avg_ep_reward', self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log('avg_reward', self.avg_reward, prog_bar=True, on_step=False, on_epoch=True)
        (optimizer_actor, optimizer_critic) = self.optimizers()
        loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
        self.manual_backward(loss_actor)
        optimizer_actor.step()
        optimizer_actor.zero_grad()
        loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
        self.manual_backward(loss_critic)
        optimizer_critic.step()
        optimizer_critic.zero_grad()
        self.log('loss_critic', loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_actor', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> List[Optimizer]:
        if False:
            for i in range(10):
                print('nop')
        'Initialize Adam optimizer.'
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        return (optimizer_actor, optimizer_critic)

    def optimizer_step(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic for each data sample."
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        if False:
            print('Hello World!')
        'Initialize the Replay Buffer dataset used for retrieving experiences.'
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        if False:
            print('Hello World!')
        'Get train loader.'
        return self._dataloader()

def main(args) -> None:
    if False:
        print('Hello World!')
    model = PPOLightning(**vars(args))
    trainer = Trainer(accelerator='cpu', devices=1, val_check_interval=100)
    trainer.fit(model)
if __name__ == '__main__':
    cli_lightning_logo()
    seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='advantage discount factor')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='learning rate of actor network')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='learning rate of critic network')
    parser.add_argument('--max_episode_len', type=int, default=1000, help='capacity of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size when training network')
    parser.add_argument('--steps_per_epoch', type=int, default=2048, help='how many action-state pairs to rollout for trajectory collection per epoch')
    parser.add_argument('--nb_optim_iters', type=int, default=4, help='how many steps of gradient descent to perform on each batch')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='hyperparameter for clipping in the policy objective')
    args = parser.parse_args()
    main(args)