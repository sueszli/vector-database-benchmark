import tensorflow as tf
import numpy as np
from gym.spaces.box import Box
from stable_baselines.common.math_util import discount_with_boundaries, scale_action, unscale_action

def test_discount_with_boundaries():
    if False:
        while True:
            i = 10
    '\n    test the discount_with_boundaries function\n    '
    gamma = 0.9
    rewards = np.array([1.0, 2.0, 3.0, 4.0], 'float32')
    episode_starts = [1.0, 0.0, 0.0, 1.0]
    discounted_rewards = discount_with_boundaries(rewards, episode_starts, gamma)
    assert np.allclose(discounted_rewards, [1 + gamma * 2 + gamma ** 2 * 3, 2 + gamma * 3, 3, 4])
    return

def test_scaling_action():
    if False:
        while True:
            i = 10
    '\n    test scaling of scalar, 1d and 2d vectors of finite non-NaN real numbers to and from tanh co-domain (per component)\n    '
    test_ranges = [(-1, 1), (-10, 10), (-10, 5), (-10, 0), (-10, -5), (0, 10), (5, 10)]
    for (range_low, range_high) in test_ranges:
        check_scaled_actions_from_range(range_low, range_high, scalar=True)
    for test_range in test_ranges:
        check_scaled_actions_from_range(*test_range)
    for (r1_low, r1_high) in test_ranges:
        for (r2_low, r2_high) in test_ranges:
            check_scaled_actions_from_range(np.array([r1_low, r2_low], dtype=np.float), np.array([r1_high, r2_high], dtype=np.float))

def check_scaled_actions_from_range(low, high, scalar=False):
    if False:
        print('Hello World!')
    '\n    helper method which creates dummy action space spanning between respective components of low and high\n    and then checks scaling to and from tanh co-domain for low, middle and high value from  that action space\n    :param low: (np.ndarray), (int) or (float)\n    :param high: (np.ndarray), (int) or (float)\n    :param scalar: (bool) Whether consider scalar range or wrap it into 1d vector\n    '
    if scalar and (isinstance(low, float) or isinstance(low, int)):
        ones = 1.0
        action_space = Box(low, high, shape=(1,))
    else:
        low = np.atleast_1d(low)
        high = np.atleast_1d(high)
        ones = np.ones_like(low)
        action_space = Box(low, high)
    mid = 0.5 * (low + high)
    expected_mapping = [(low, -ones), (mid, 0.0 * ones), (high, ones)]
    for (not_scaled, scaled) in expected_mapping:
        assert np.allclose(scale_action(action_space, not_scaled), scaled)
        assert np.allclose(unscale_action(action_space, scaled), not_scaled)

def test_batch_shape_invariant_to_scaling():
    if False:
        i = 10
        return i + 15
    '\n    test that scaling deals well with batches as tensors and numpy matrices in terms of shape\n    '
    action_space = Box(np.array([-10.0, -5.0, -1.0]), np.array([10.0, 3.0, 2.0]))
    tensor = tf.constant(1.0, shape=[2, 3])
    matrix = np.ones((2, 3))
    assert scale_action(action_space, tensor).shape == (2, 3)
    assert scale_action(action_space, matrix).shape == (2, 3)
    assert unscale_action(action_space, tensor).shape == (2, 3)
    assert unscale_action(action_space, matrix).shape == (2, 3)