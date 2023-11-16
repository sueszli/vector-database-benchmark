"""Reward shaping functions used by Contexts.

  Each reward function should take the following inputs and return new rewards,
    and discounts.

  new_rewards, discounts = reward_fn(states, actions, rewards,
    next_states, contexts)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin.tf

def summarize_stats(stats):
    if False:
        for i in range(10):
            print('nop')
    'Summarize a dictionary of variables.\n\n  Args:\n    stats: a dictionary of {name: tensor} to compute stats over.\n  '
    for (name, stat) in stats.items():
        mean = tf.reduce_mean(stat)
        tf.summary.scalar('mean_%s' % name, mean)
        tf.summary.scalar('max_%s' % name, tf.reduce_max(stat))
        tf.summary.scalar('min_%s' % name, tf.reduce_min(stat))
        std = tf.sqrt(tf.reduce_mean(tf.square(stat)) - tf.square(mean) + 1e-10)
        tf.summary.scalar('std_%s' % name, std)
        tf.summary.histogram(name, stat)

def index_states(states, indices):
    if False:
        print('Hello World!')
    'Return indexed states.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    indices: (a list of Numpy integer array) Indices of states dimensions\n      to be mapped.\n  Returns:\n    A [batch_size, num_indices] Tensor representing the batch of indexed states.\n  '
    if indices is None:
        return states
    indices = tf.constant(indices, dtype=tf.int32)
    return tf.gather(states, indices=indices, axis=1)

def record_tensor(tensor, indices, stats, name='states'):
    if False:
        while True:
            i = 10
    'Record specified tensor dimensions into stats.\n\n  Args:\n    tensor: A [batch_size, num_dims] Tensor.\n    indices: (a list of integers) Indices of dimensions to record.\n    stats: A dictionary holding stats.\n    name: (string) Name of tensor.\n  '
    if indices is None:
        indices = range(tensor.shape.as_list()[1])
    for index in indices:
        stats['%s_%02d' % (name, index)] = tensor[:, index]

@gin.configurable
def potential_rewards(states, actions, rewards, next_states, contexts, gamma=1.0, reward_fn=None):
    if False:
        for i in range(10):
            print('nop')
    'Return the potential-based rewards.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    gamma: Reward discount.\n    reward_fn: A reward function.\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del actions
    gamma = tf.to_float(gamma)
    (rewards_tp1, discounts) = reward_fn(None, None, rewards, next_states, contexts)
    (rewards, _) = reward_fn(None, None, rewards, states, contexts)
    return (-rewards + gamma * rewards_tp1, discounts)

@gin.configurable
def timed_rewards(states, actions, rewards, next_states, contexts, reward_fn=None, dense=False, timer_index=-1):
    if False:
        print('Hello World!')
    'Return the timed rewards.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    reward_fn: A reward function.\n    dense: (boolean) Provide dense rewards or sparse rewards at time = 0.\n    timer_index: (integer) The context list index that specifies timer.\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    assert contexts[timer_index].get_shape().as_list()[1] == 1
    timers = contexts[timer_index][:, 0]
    (rewards, discounts) = reward_fn(states, actions, rewards, next_states, contexts)
    terminates = tf.to_float(timers <= 0)
    for _ in range(rewards.shape.ndims - 1):
        terminates = tf.expand_dims(terminates, axis=-1)
    if not dense:
        rewards *= terminates
    discounts *= tf.to_float(1.0) - terminates
    return (rewards, discounts)

@gin.configurable
def reset_rewards(states, actions, rewards, next_states, contexts, reset_index=0, reset_state=None, reset_reward_function=None, include_forward_rewards=True, include_reset_rewards=True):
    if False:
        i = 10
        return i + 15
    'Returns the rewards for a forward/reset agent.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    reset_index: (integer) The context list index that specifies reset.\n    reset_state: Reset state.\n    reset_reward_function: Reward function for reset step.\n    include_forward_rewards: Include the rewards from the forward pass.\n    include_reset_rewards: Include the rewards from the reset pass.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    reset_state = tf.constant(reset_state, dtype=next_states.dtype, shape=next_states.shape)
    reset_states = tf.expand_dims(reset_state, 0)

    def true_fn():
        if False:
            print('Hello World!')
        if include_reset_rewards:
            return reset_reward_function(states, actions, rewards, next_states, [reset_states] + contexts[1:])
        else:
            return (tf.zeros_like(rewards), tf.ones_like(rewards))

    def false_fn():
        if False:
            while True:
                i = 10
        if include_forward_rewards:
            return plain_rewards(states, actions, rewards, next_states, contexts)
        else:
            return (tf.zeros_like(rewards), tf.ones_like(rewards))
    (rewards, discounts) = tf.cond(tf.cast(contexts[reset_index][0, 0], dtype=tf.bool), true_fn, false_fn)
    return (rewards, discounts)

@gin.configurable
def tanh_similarity(states, actions, rewards, next_states, contexts, mse_scale=1.0, state_scales=1.0, goal_scales=1.0, summarize=False):
    if False:
        i = 10
        return i + 15
    'Returns the similarity between next_states and contexts using tanh and mse.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    mse_scale: A float, to scale mse before tanh.\n    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,\n      must be broadcastable to number of state dimensions.\n    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,\n      must be broadcastable to number of goal dimensions.\n    summarize: (boolean) enable summary ops.\n\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, actions, rewards
    mse = tf.reduce_mean(tf.squared_difference(next_states * state_scales, contexts[0] * goal_scales), -1)
    tanh = tf.tanh(mse_scale * mse)
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_mse', tf.reduce_mean(mse))
            tf.summary.histogram('mse', mse)
            tf.summary.scalar('mean_tanh', tf.reduce_mean(tanh))
            tf.summary.histogram('tanh', tanh)
    rewards = tf.to_float(1 - tanh)
    return (rewards, tf.ones_like(rewards))

@gin.configurable
def negative_mse(states, actions, rewards, next_states, contexts, state_scales=1.0, goal_scales=1.0, summarize=False):
    if False:
        while True:
            i = 10
    'Returns the negative mean square error between next_states and contexts.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,\n      must be broadcastable to number of state dimensions.\n    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,\n      must be broadcastable to number of goal dimensions.\n    summarize: (boolean) enable summary ops.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, actions, rewards
    mse = tf.reduce_mean(tf.squared_difference(next_states * state_scales, contexts[0] * goal_scales), -1)
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_mse', tf.reduce_mean(mse))
            tf.summary.histogram('mse', mse)
    rewards = tf.to_float(-mse)
    return (rewards, tf.ones_like(rewards))

@gin.configurable
def negative_distance(states, actions, rewards, next_states, contexts, state_scales=1.0, goal_scales=1.0, reward_scales=1.0, weight_index=None, weight_vector=None, summarize=False, termination_epsilon=0.0001, state_indices=None, goal_indices=None, vectorize=False, relative_context=False, diff=False, norm='L2', epsilon=1e-10, bonus_epsilon=0.0, offset=0.0):
    if False:
        while True:
            i = 10
    'Returns the negative euclidean distance between next_states and contexts.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,\n      must be broadcastable to number of state dimensions.\n    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,\n      must be broadcastable to number of goal dimensions.\n    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,\n      must be broadcastable to number of reward dimensions.\n    weight_index: (integer) The context list index that specifies weight.\n    weight_vector: (a number or a list or Numpy array) The weighting vector,\n      broadcastable to `next_states`.\n    summarize: (boolean) enable summary ops.\n    termination_epsilon: terminate if dist is less than this quantity.\n    state_indices: (a list of integers) list of state indices to select.\n    goal_indices: (a list of integers) list of goal indices to select.\n    vectorize: Return a vectorized form.\n    norm: L1 or L2.\n    epsilon: small offset to ensure non-negative/zero distance.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del actions, rewards
    stats = {}
    record_tensor(next_states, state_indices, stats, 'next_states')
    states = index_states(states, state_indices)
    next_states = index_states(next_states, state_indices)
    goals = index_states(contexts[0], goal_indices)
    if relative_context:
        goals = states + goals
    sq_dists = tf.squared_difference(next_states * state_scales, goals * goal_scales)
    old_sq_dists = tf.squared_difference(states * state_scales, goals * goal_scales)
    record_tensor(sq_dists, None, stats, 'sq_dists')
    if weight_vector is not None:
        sq_dists *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
        old_sq_dists *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
    if weight_index is not None:
        weights = tf.abs(index_states(contexts[0], weight_index))
        sq_dists *= weights
        old_sq_dists *= weights
    if norm == 'L1':
        dist = tf.sqrt(sq_dists + epsilon)
        old_dist = tf.sqrt(old_sq_dists + epsilon)
        if not vectorize:
            dist = tf.reduce_sum(dist, -1)
            old_dist = tf.reduce_sum(old_dist, -1)
    elif norm == 'L2':
        if vectorize:
            dist = sq_dists
            old_dist = old_sq_dists
        else:
            dist = tf.reduce_sum(sq_dists, -1)
            old_dist = tf.reduce_sum(old_sq_dists, -1)
        dist = tf.sqrt(dist + epsilon)
        old_dist = tf.sqrt(old_dist + epsilon)
    else:
        raise NotImplementedError(norm)
    discounts = dist > termination_epsilon
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_dist', tf.reduce_mean(dist))
            tf.summary.histogram('dist', dist)
            summarize_stats(stats)
    bonus = tf.to_float(dist < bonus_epsilon)
    dist *= reward_scales
    old_dist *= reward_scales
    if diff:
        return (bonus + offset + tf.to_float(old_dist - dist), tf.to_float(discounts))
    return (bonus + offset + tf.to_float(-dist), tf.to_float(discounts))

@gin.configurable
def cosine_similarity(states, actions, rewards, next_states, contexts, state_scales=1.0, goal_scales=1.0, reward_scales=1.0, normalize_states=True, normalize_goals=True, weight_index=None, weight_vector=None, summarize=False, state_indices=None, goal_indices=None, offset=0.0):
    if False:
        i = 10
        return i + 15
    'Returns the cosine similarity between next_states - states and contexts.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,\n      must be broadcastable to number of state dimensions.\n    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,\n      must be broadcastable to number of goal dimensions.\n    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,\n      must be broadcastable to number of reward dimensions.\n    weight_index: (integer) The context list index that specifies weight.\n    weight_vector: (a number or a list or Numpy array) The weighting vector,\n      broadcastable to `next_states`.\n    summarize: (boolean) enable summary ops.\n    termination_epsilon: terminate if dist is less than this quantity.\n    state_indices: (a list of integers) list of state indices to select.\n    goal_indices: (a list of integers) list of goal indices to select.\n    vectorize: Return a vectorized form.\n    norm: L1 or L2.\n    epsilon: small offset to ensure non-negative/zero distance.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del actions, rewards
    stats = {}
    record_tensor(next_states, state_indices, stats, 'next_states')
    states = index_states(states, state_indices)
    next_states = index_states(next_states, state_indices)
    goals = index_states(contexts[0], goal_indices)
    if weight_vector is not None:
        goals *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
    if weight_index is not None:
        weights = tf.abs(index_states(contexts[0], weight_index))
        goals *= weights
    direction_vec = next_states - states
    if normalize_states:
        direction_vec = tf.nn.l2_normalize(direction_vec, -1)
    goal_vec = goals
    if normalize_goals:
        goal_vec = tf.nn.l2_normalize(goal_vec, -1)
    similarity = tf.reduce_sum(goal_vec * direction_vec, -1)
    discounts = tf.ones_like(similarity)
    return (offset + tf.to_float(similarity), tf.to_float(discounts))

@gin.configurable
def diff_distance(states, actions, rewards, next_states, contexts, state_scales=1.0, goal_scales=1.0, reward_scales=1.0, weight_index=None, weight_vector=None, summarize=False, termination_epsilon=0.0001, state_indices=None, goal_indices=None, norm='L2', epsilon=1e-10):
    if False:
        for i in range(10):
            print('nop')
    'Returns the difference in euclidean distance between states/next_states and contexts.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,\n      must be broadcastable to number of state dimensions.\n    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,\n      must be broadcastable to number of goal dimensions.\n    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,\n      must be broadcastable to number of reward dimensions.\n    weight_index: (integer) The context list index that specifies weight.\n    weight_vector: (a number or a list or Numpy array) The weighting vector,\n      broadcastable to `next_states`.\n    summarize: (boolean) enable summary ops.\n    termination_epsilon: terminate if dist is less than this quantity.\n    state_indices: (a list of integers) list of state indices to select.\n    goal_indices: (a list of integers) list of goal indices to select.\n    vectorize: Return a vectorized form.\n    norm: L1 or L2.\n    epsilon: small offset to ensure non-negative/zero distance.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del actions, rewards
    stats = {}
    record_tensor(next_states, state_indices, stats, 'next_states')
    next_states = index_states(next_states, state_indices)
    states = index_states(states, state_indices)
    goals = index_states(contexts[0], goal_indices)
    next_sq_dists = tf.squared_difference(next_states * state_scales, goals * goal_scales)
    sq_dists = tf.squared_difference(states * state_scales, goals * goal_scales)
    record_tensor(sq_dists, None, stats, 'sq_dists')
    if weight_vector is not None:
        next_sq_dists *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
        sq_dists *= tf.convert_to_tensor(weight_vector, dtype=next_states.dtype)
    if weight_index is not None:
        next_sq_dists *= contexts[weight_index]
        sq_dists *= contexts[weight_index]
    if norm == 'L1':
        next_dist = tf.sqrt(next_sq_dists + epsilon)
        dist = tf.sqrt(sq_dists + epsilon)
        next_dist = tf.reduce_sum(next_dist, -1)
        dist = tf.reduce_sum(dist, -1)
    elif norm == 'L2':
        next_dist = tf.reduce_sum(next_sq_dists, -1)
        next_dist = tf.sqrt(next_dist + epsilon)
        dist = tf.reduce_sum(sq_dists, -1)
        dist = tf.sqrt(dist + epsilon)
    else:
        raise NotImplementedError(norm)
    discounts = next_dist > termination_epsilon
    if summarize:
        with tf.name_scope('RewardFn/'):
            tf.summary.scalar('mean_dist', tf.reduce_mean(dist))
            tf.summary.histogram('dist', dist)
            summarize_stats(stats)
    diff = dist - next_dist
    diff *= reward_scales
    return (tf.to_float(diff), tf.to_float(discounts))

@gin.configurable
def binary_indicator(states, actions, rewards, next_states, contexts, termination_epsilon=0.0001, offset=0, epsilon=1e-10, state_indices=None, summarize=False):
    if False:
        while True:
            i = 10
    'Returns 0/1 by checking if next_states and contexts overlap.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    termination_epsilon: terminate if dist is less than this quantity.\n    offset: Offset the rewards.\n    epsilon: small offset to ensure non-negative/zero distance.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, actions
    next_states = index_states(next_states, state_indices)
    dist = tf.reduce_sum(tf.squared_difference(next_states, contexts[0]), -1)
    dist = tf.sqrt(dist + epsilon)
    discounts = dist > termination_epsilon
    rewards = tf.logical_not(discounts)
    rewards = tf.to_float(rewards) + offset
    return (tf.to_float(rewards), tf.ones_like(tf.to_float(discounts)))

@gin.configurable
def plain_rewards(states, actions, rewards, next_states, contexts):
    if False:
        i = 10
        return i + 15
    'Returns the given rewards.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, actions, next_states, contexts
    return (rewards, tf.ones_like(rewards))

@gin.configurable
def ctrl_rewards(states, actions, rewards, next_states, contexts, reward_scales=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Returns the negative control cost.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,\n      must be broadcastable to number of reward dimensions.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, rewards, contexts
    if actions is None:
        rewards = tf.to_float(tf.zeros(shape=next_states.shape[:1]))
    else:
        rewards = -tf.reduce_sum(tf.square(actions), axis=1)
        rewards *= reward_scales
        rewards = tf.to_float(rewards)
    return (rewards, tf.ones_like(rewards))

@gin.configurable
def diff_rewards(states, actions, rewards, next_states, contexts, state_indices=None, goal_index=0):
    if False:
        while True:
            i = 10
    'Returns (next_states - goals) as a batched vector reward.'
    del states, rewards, actions
    if state_indices is not None:
        next_states = index_states(next_states, state_indices)
    rewards = tf.to_float(next_states - contexts[goal_index])
    return (rewards, tf.ones_like(rewards))

@gin.configurable
def state_rewards(states, actions, rewards, next_states, contexts, weight_index=None, state_indices=None, weight_vector=1.0, offset_vector=0.0, summarize=False):
    if False:
        print('Hello World!')
    'Returns the rewards that are linear mapping of next_states.\n\n  Args:\n    states: A [batch_size, num_state_dims] Tensor representing a batch\n        of states.\n    actions: A [batch_size, num_action_dims] Tensor representing a batch\n      of actions.\n    rewards: A [batch_size] Tensor representing a batch of rewards.\n    next_states: A [batch_size, num_state_dims] Tensor representing a batch\n      of next states.\n    contexts: A list of [batch_size, num_context_dims] Tensor representing\n      a batch of contexts.\n    weight_index: (integer) Index of contexts lists that specify weighting.\n    state_indices: (a list of Numpy integer array) Indices of states dimensions\n      to be mapped.\n    weight_vector: (a number or a list or Numpy array) The weighting vector,\n      broadcastable to `next_states`.\n    offset_vector: (a number or a list of Numpy array) The off vector.\n    summarize: (boolean) enable summary ops.\n\n  Returns:\n    A new tf.float32 [batch_size] rewards Tensor, and\n      tf.float32 [batch_size] discounts tensor.\n  '
    del states, actions, rewards
    stats = {}
    record_tensor(next_states, state_indices, stats)
    next_states = index_states(next_states, state_indices)
    weight = tf.constant(weight_vector, dtype=next_states.dtype, shape=next_states[0].shape)
    weights = tf.expand_dims(weight, 0)
    offset = tf.constant(offset_vector, dtype=next_states.dtype, shape=next_states[0].shape)
    offsets = tf.expand_dims(offset, 0)
    if weight_index is not None:
        weights *= contexts[weight_index]
    rewards = tf.to_float(tf.reduce_sum(weights * (next_states + offsets), axis=1))
    if summarize:
        with tf.name_scope('RewardFn/'):
            summarize_stats(stats)
    return (rewards, tf.ones_like(rewards))