import gymnasium as gym
import numpy as np
import pytest
import torch as th
from torch.distributions import Normal
from stable_baselines3 import A2C, PPO, SAC

def test_state_dependent_exploration_grad():
    if False:
        print('Hello World!')
    '\n    Check that the gradient correspond to the expected one\n    '
    n_states = 2
    state_dim = 3
    action_dim = 10
    sigma_hat = th.ones(state_dim, action_dim, requires_grad=True)
    th.manual_seed(2)
    weights_dist = Normal(th.zeros_like(sigma_hat), sigma_hat)
    weights = weights_dist.rsample()
    state = th.rand(n_states, state_dim)
    mu = th.ones(action_dim)
    noise = th.mm(state, weights)
    action = mu + noise
    variance = th.mm(state ** 2, sigma_hat ** 2)
    action_dist = Normal(mu, th.sqrt(variance))
    loss = action_dist.log_prob(action.detach()).sum(dim=-1).mean()
    loss.backward()
    grad = th.zeros_like(sigma_hat)
    for j in range(action_dim):
        sigma_j = th.sqrt(variance[:, j])
        for i in range(state_dim):
            d_log_policy_j = (noise[:, j] ** 2 - sigma_j ** 2) / sigma_j ** 3
            d_log_sigma_j = state[:, i] ** 2 * sigma_hat[i, j] / sigma_j
            grad[i, j] = (d_log_policy_j * d_log_sigma_j).mean()
    assert sigma_hat.grad.allclose(grad)

def test_sde_check():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        PPO('MlpPolicy', 'CartPole-v1', use_sde=True)

def test_only_sde_squashed():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError, match='use_sde=True'):
        PPO('MlpPolicy', 'Pendulum-v1', use_sde=False, policy_kwargs=dict(squash_output=True))

@pytest.mark.parametrize('model_class', [SAC, A2C, PPO])
@pytest.mark.parametrize('use_expln', [False, True])
@pytest.mark.parametrize('squash_output', [False, True])
def test_state_dependent_noise(model_class, use_expln, squash_output):
    if False:
        print('Hello World!')
    kwargs = {'learning_starts': 0} if model_class == SAC else {'n_steps': 64}
    policy_kwargs = dict(log_std_init=-2, use_expln=use_expln, net_arch=[64])
    if model_class in [A2C, PPO]:
        policy_kwargs['squash_output'] = squash_output
    elif not squash_output:
        pytest.skip('SAC can only use squashed output')
    env = StoreActionEnvWrapper(gym.make('Pendulum-v1'))
    model = model_class('MlpPolicy', env, use_sde=True, seed=1, verbose=1, policy_kwargs=policy_kwargs, **kwargs)
    model.learn(total_timesteps=255)
    buffer = model.replay_buffer if model_class == SAC else model.rollout_buffer
    assert (buffer.actions <= model.action_space.high).all()
    assert (buffer.actions >= model.action_space.low).all()
    if squash_output:
        if buffer.actions.max() > 0.5:
            assert np.max(env.actions) > 1.0
        if buffer.actions.max() < -0.5:
            assert np.min(env.actions) < -1.0
    model.policy.reset_noise()
    if model_class == SAC:
        model.policy.actor.get_std()

class StoreActionEnvWrapper(gym.Wrapper):
    """
    Keep track of which actions were sent to the env.
    """

    def __init__(self, env):
        if False:
            while True:
                i = 10
        super().__init__(env)
        self.actions = []

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        self.actions.append(action)
        return super().step(action)