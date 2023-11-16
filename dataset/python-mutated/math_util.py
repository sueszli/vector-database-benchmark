import numpy as np
import scipy.signal

def safe_mean(arr):
    if False:
        i = 10
        return i + 15
    '\n    Compute the mean of an array if there is at least one element.\n    For empty array, return nan. It is used for logging only.\n\n    :param arr: (np.ndarray)\n    :return: (float)\n    '
    return np.nan if len(arr) == 0 else np.mean(arr)

def discount(vector, gamma):
    if False:
        return 10
    '\n    computes discounted sums along 0th dimension of vector x.\n        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],\n                where k = len(x) - t - 1\n\n    :param vector: (np.ndarray) the input vector\n    :param gamma: (float) the discount value\n    :return: (np.ndarray) the output vector\n    '
    assert vector.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], vector[::-1], axis=0)[::-1]

def explained_variance(y_pred, y_true):
    if False:
        print('Hello World!')
    '\n    Computes fraction of variance that ypred explains about y.\n    Returns 1 - Var[y-ypred] / Var[y]\n\n    interpretation:\n        ev=0  =>  might as well have predicted zero\n        ev=1  =>  perfect prediction\n        ev<0  =>  worse than just predicting zero\n\n    :param y_pred: (np.ndarray) the prediction\n    :param y_true: (np.ndarray) the expected value\n    :return: (float) explained variance of ypred and y\n    '
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def explained_variance_2d(y_pred, y_true):
    if False:
        while True:
            i = 10
    '\n    Computes fraction of variance that ypred explains about y, for 2D arrays.\n    Returns 1 - Var[y-ypred] / Var[y]\n\n    interpretation:\n        ev=0  =>  might as well have predicted zero\n        ev=1  =>  perfect prediction\n        ev<0  =>  worse than just predicting zero\n\n    :param y_pred: (np.ndarray) the prediction\n    :param y_true: (np.ndarray) the expected value\n    :return: (float) explained variance of ypred and y\n    '
    assert y_true.ndim == 2 and y_pred.ndim == 2
    var_y = np.var(y_true, axis=0)
    explained_var = 1 - np.var(y_true - y_pred) / var_y
    explained_var[var_y < 1e-10] = 0
    return explained_var

def flatten_arrays(arrs):
    if False:
        while True:
            i = 10
    '\n    flattens a list of arrays down to 1D\n\n    :param arrs: ([np.ndarray]) arrays\n    :return: (np.ndarray) 1D flattened array\n    '
    return np.concatenate([arr.flat for arr in arrs])

def unflatten_vector(vec, shapes):
    if False:
        return 10
    '\n    reshape a flattened array\n\n    :param vec: (np.ndarray) 1D arrays\n    :param shapes: (tuple)\n    :return: ([np.ndarray]) reshaped array\n    '
    i = 0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i + size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs

def discount_with_boundaries(rewards, episode_starts, gamma):
    if False:
        while True:
            i = 10
    '\n    computes discounted sums along 0th dimension of x (reward), while taking into account the start of each episode.\n        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],\n                where k = len(x) - t - 1\n\n    :param rewards: (np.ndarray) the input vector (rewards)\n    :param episode_starts: (np.ndarray) 2d array of bools, indicating when a new episode has started\n    :param gamma: (float) the discount factor\n    :return: (np.ndarray) the output vector (discounted rewards)\n    '
    discounted_rewards = np.zeros_like(rewards)
    n_samples = rewards.shape[0]
    discounted_rewards[n_samples - 1] = rewards[n_samples - 1]
    for step in range(n_samples - 2, -1, -1):
        discounted_rewards[step] = rewards[step] + gamma * discounted_rewards[step + 1] * (1 - episode_starts[step + 1])
    return discounted_rewards

def scale_action(action_space, action):
    if False:
        print('Hello World!')
    '\n    Rescale the action from [low, high] to [-1, 1]\n    (no need for symmetric action space)\n\n    :param action_space: (gym.spaces.box.Box)\n    :param action: (np.ndarray)\n    :return: (np.ndarray)\n    '
    (low, high) = (action_space.low, action_space.high)
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unscale_action(action_space, scaled_action):
    if False:
        while True:
            i = 10
    '\n    Rescale the action from [-1, 1] to [low, high]\n    (no need for symmetric action space)\n\n    :param action_space: (gym.spaces.box.Box)\n    :param action: (np.ndarray)\n    :return: (np.ndarray)\n    '
    (low, high) = (action_space.low, action_space.high)
    return low + 0.5 * (scaled_action + 1.0) * (high - low)