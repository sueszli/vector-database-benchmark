"""Several functions to sample contextual data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def sample_contextual_data(num_contexts, dim_context, num_actions, sigma):
    if False:
        return 10
    'Samples independent Gaussian data.\n\n  There is nothing to learn here as the rewards do not depend on the context.\n\n  Args:\n    num_contexts: Number of contexts to sample.\n    dim_context: Dimension of the contexts.\n    num_actions: Number of arms for the multi-armed bandit.\n    sigma: Standard deviation of the independent Gaussian samples.\n\n  Returns:\n    data: A [num_contexts, dim_context + num_actions] numpy array with the data.\n  '
    size_data = [num_contexts, dim_context + num_actions]
    return np.random.normal(scale=sigma, size=size_data)

def sample_linear_data(num_contexts, dim_context, num_actions, sigma=0.0):
    if False:
        i = 10
        return i + 15
    "Samples data from linearly parameterized arms.\n\n  The reward for context X and arm j is given by X^T beta_j, for some latent\n  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly\n  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.\n\n  Args:\n    num_contexts: Number of contexts to sample.\n    dim_context: Dimension of the contexts.\n    num_actions: Number of arms for the multi-armed bandit.\n    sigma: Standard deviation of the additive noise. Set to zero for no noise.\n\n  Returns:\n    data: A [n, d+k] numpy array with the data.\n    betas: Latent parameters that determine expected reward for each arm.\n    opt: (optimal_rewards, optimal_actions) for all contexts.\n  "
    betas = np.random.uniform(-1, 1, (dim_context, num_actions))
    betas /= np.linalg.norm(betas, axis=0)
    contexts = np.random.normal(size=[num_contexts, dim_context])
    rewards = np.dot(contexts, betas)
    opt_actions = np.argmax(rewards, axis=1)
    rewards += np.random.normal(scale=sigma, size=rewards.shape)
    opt_rewards = np.array([rewards[i, act] for (i, act) in enumerate(opt_actions)])
    return (np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions))

def sample_sparse_linear_data(num_contexts, dim_context, num_actions, sparse_dim, sigma=0.0):
    if False:
        return 10
    "Samples data from sparse linearly parameterized arms.\n\n  The reward for context X and arm j is given by X^T beta_j, for some latent\n  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly\n  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.\n  Only s components out of d are non-zero for each arm's beta.\n\n  Args:\n    num_contexts: Number of contexts to sample.\n    dim_context: Dimension of the contexts.\n    num_actions: Number of arms for the multi-armed bandit.\n    sparse_dim: Dimension of the latent subspace (sparsity pattern dimension).\n    sigma: Standard deviation of the additive noise. Set to zero for no noise.\n\n  Returns:\n    data: A [num_contexts, dim_context+num_actions] numpy array with the data.\n    betas: Latent parameters that determine expected reward for each arm.\n    opt: (optimal_rewards, optimal_actions) for all contexts.\n  "
    flatten = lambda l: [item for sublist in l for item in sublist]
    sparse_pattern = flatten([[(j, i) for j in np.random.choice(range(dim_context), sparse_dim, replace=False)] for i in range(num_actions)])
    betas = np.random.uniform(-1, 1, (dim_context, num_actions))
    mask = np.zeros((dim_context, num_actions))
    for elt in sparse_pattern:
        mask[elt] = 1
    betas = np.multiply(betas, mask)
    betas /= np.linalg.norm(betas, axis=0)
    contexts = np.random.normal(size=[num_contexts, dim_context])
    rewards = np.dot(contexts, betas)
    opt_actions = np.argmax(rewards, axis=1)
    rewards += np.random.normal(scale=sigma, size=rewards.shape)
    opt_rewards = np.array([rewards[i, act] for (i, act) in enumerate(opt_actions)])
    return (np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions))

def sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v, mu_large, std_large):
    if False:
        return 10
    'Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).\n\n  Args:\n    num_contexts: Number of points to sample, i.e. (context, action rewards).\n    delta: Exploration parameter: high reward in one region if norm above delta.\n    mean_v: Mean reward for each action if context norm is below delta.\n    std_v: Gaussian reward std for each action if context norm is below delta.\n    mu_large: Mean reward for optimal action if context norm is above delta.\n    std_large: Reward std for optimal action if context norm is above delta.\n\n  Returns:\n    dataset: Sampled matrix with n rows: (context, action rewards).\n    opt_vals: Vector of expected optimal (reward, action) for each context.\n  '
    context_dim = 2
    num_actions = 5
    data = []
    rewards = []
    opt_actions = []
    opt_rewards = []
    while len(data) < num_contexts:
        raw_data = np.random.uniform(-1, 1, (int(num_contexts / 3), context_dim))
        for i in range(raw_data.shape[0]):
            if np.linalg.norm(raw_data[i, :]) <= 1:
                data.append(raw_data[i, :])
    contexts = np.stack(data)[:num_contexts, :]
    for i in range(num_contexts):
        r = [np.random.normal(mean_v[j], std_v[j]) for j in range(num_actions)]
        if np.linalg.norm(contexts[i, :]) >= delta:
            r_big = np.random.normal(mu_large, std_large)
            if contexts[i, 0] > 0:
                if contexts[i, 1] > 0:
                    r[0] = r_big
                    opt_actions.append(0)
                else:
                    r[1] = r_big
                    opt_actions.append(1)
            elif contexts[i, 1] > 0:
                r[2] = r_big
                opt_actions.append(2)
            else:
                r[3] = r_big
                opt_actions.append(3)
        else:
            opt_actions.append(np.argmax(mean_v))
        opt_rewards.append(r[opt_actions[-1]])
        rewards.append(r)
    rewards = np.stack(rewards)
    opt_rewards = np.array(opt_rewards)
    opt_actions = np.array(opt_actions)
    return (np.hstack((contexts, rewards)), (opt_rewards, opt_actions))