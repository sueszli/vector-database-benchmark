import numpy as np
import tensorflow as tf
from six.moves import xrange
__all__ = ['discount_episode_rewards', 'cross_entropy_reward_loss', 'log_weight', 'choice_action_by_probs']

def discount_episode_rewards(rewards=None, gamma=0.99, mode=0):
    if False:
        for i in range(10):
            print('nop')
    'Take 1D float array of rewards and compute discounted rewards for an\n    episode. When encount a non-zero value, consider as the end a of an episode.\n\n    Parameters\n    ----------\n    rewards : list\n        List of rewards\n    gamma : float\n        Discounted factor\n    mode : int\n        Mode for computing the discount rewards.\n            - If mode == 0, reset the discount process when encount a non-zero reward (Ping-pong game).\n            - If mode == 1, would not reset the discount process.\n\n    Returns\n    --------\n    list of float\n        The discounted rewards.\n\n    Examples\n    ----------\n    >>> rewards = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])\n    >>> gamma = 0.9\n    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma)\n    >>> print(discount_rewards)\n    [ 0.72899997  0.81        0.89999998  1.          0.72899997  0.81\n    0.89999998  1.          0.72899997  0.81        0.89999998  1.        ]\n    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma, mode=1)\n    >>> print(discount_rewards)\n    [ 1.52110755  1.69011939  1.87791049  2.08656716  1.20729685  1.34144104\n    1.49048996  1.65610003  0.72899997  0.81        0.89999998  1.        ]\n\n    '
    if rewards is None:
        raise Exception('rewards should be a list')
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if mode == 0:
            if rewards[t] != 0:
                running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    if False:
        print('Hello World!')
    "Calculate the loss for Policy Gradient Network.\n\n    Parameters\n    ----------\n    logits : tensor\n        The network outputs without softmax. This function implements softmax inside.\n    actions : tensor or placeholder\n        The agent actions.\n    rewards : tensor or placeholder\n        The rewards.\n\n    Returns\n    --------\n    Tensor\n        The TensorFlow loss function.\n\n    Examples\n    ----------\n    >>> states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])\n    >>> network = InputLayer(states_batch_pl, name='input')\n    >>> network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='relu1')\n    >>> network = DenseLayer(network, n_units=3, name='out')\n    >>> probs = network.outputs\n    >>> sampling_prob = tf.nn.softmax(probs)\n    >>> actions_batch_pl = tf.placeholder(tf.int32, shape=[None])\n    >>> discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])\n    >>> loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl, discount_rewards_batch_pl)\n    >>> train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)\n\n    "
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits, name=name)
    return tf.reduce_sum(tf.multiply(cross_entropy, rewards))

def log_weight(probs, weights, name='log_weight'):
    if False:
        for i in range(10):
            print('nop')
    'Log weight.\n\n    Parameters\n    -----------\n    probs : tensor\n        If it is a network output, usually we should scale it to [0, 1] via softmax.\n    weights : tensor\n        The weights.\n\n    Returns\n    --------\n    Tensor\n        The Tensor after appling the log weighted expression.\n\n    '
    with tf.variable_scope(name):
        exp_v = tf.reduce_mean(tf.log(probs) * weights)
        return exp_v

def choice_action_by_probs(probs=(0.5, 0.5), action_list=None):
    if False:
        while True:
            i = 10
    "Choice and return an an action by given the action probability distribution.\n\n    Parameters\n    ------------\n    probs : list of float.\n        The probability distribution of all actions.\n    action_list : None or a list of int or others\n        A list of action in integer, string or others. If None, returns an integer range between 0 and len(probs)-1.\n\n    Returns\n    --------\n    float int or str\n        The chosen action.\n\n    Examples\n    ----------\n    >>> for _ in range(5):\n    >>>     a = choice_action_by_probs([0.2, 0.4, 0.4])\n    >>>     print(a)\n    0\n    1\n    1\n    2\n    1\n    >>> for _ in range(3):\n    >>>     a = choice_action_by_probs([0.5, 0.5], ['a', 'b'])\n    >>>     print(a)\n    a\n    b\n    b\n\n    "
    if action_list is None:
        n_action = len(probs)
        action_list = np.arange(n_action)
    elif len(action_list) != len(probs):
        raise Exception('number of actions should equal to number of probabilities.')
    return np.random.choice(action_list, p=probs)