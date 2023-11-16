"""Sample actor(policy) and critic(q) networks to use with DDPG/NAF agents.

The DDPG networks are defined in "Section 7: Experiment Details" of
"Continuous control with deep reinforcement learning" - Lilicrap et al.
https://arxiv.org/abs/1509.02971

The NAF critic network is based on "Section 4" of "Continuous deep Q-learning
with model-based acceleration" - Gu et al. https://arxiv.org/pdf/1603.00748.
"""
import tensorflow as tf
slim = tf.contrib.slim
import gin.tf

@gin.configurable('ddpg_critic_net')
def critic_net(states, actions, for_critic_loss=False, num_reward_dims=1, states_hidden_layers=(400,), actions_hidden_layers=None, joint_hidden_layers=(300,), weight_decay=0.0001, normalizer_fn=None, activation_fn=tf.nn.relu, zero_obs=False, images=False):
    if False:
        i = 10
        return i + 15
    'Creates a critic that returns q values for the given states and actions.\n\n  Args:\n    states: (castable to tf.float32) a [batch_size, num_state_dims] tensor\n      representing a batch of states.\n    actions: (castable to tf.float32) a [batch_size, num_action_dims] tensor\n      representing a batch of actions.\n    num_reward_dims: Number of reward dimensions.\n    states_hidden_layers: tuple of hidden layers units for states.\n    actions_hidden_layers: tuple of hidden layers units for actions.\n    joint_hidden_layers: tuple of hidden layers units after joining states\n      and actions using tf.concat().\n    weight_decay: Weight decay for l2 weights regularizer.\n    normalizer_fn: Normalizer function, i.e. slim.layer_norm,\n    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...\n  Returns:\n    A tf.float32 [batch_size] tensor of q values, or a tf.float32\n      [batch_size, num_reward_dims] tensor of vector q values if\n      num_reward_dims > 1.\n  '
    with slim.arg_scope([slim.fully_connected], activation_fn=activation_fn, normalizer_fn=normalizer_fn, weights_regularizer=slim.l2_regularizer(weight_decay), weights_initializer=slim.variance_scaling_initializer(factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):
        orig_states = tf.to_float(states)
        states = tf.concat([tf.to_float(states), tf.to_float(actions)], -1)
        if images or zero_obs:
            states *= tf.constant([0.0] * 2 + [1.0] * (states.shape[1] - 2))
        actions = tf.to_float(actions)
        if states_hidden_layers:
            states = slim.stack(states, slim.fully_connected, states_hidden_layers, scope='states')
        if actions_hidden_layers:
            actions = slim.stack(actions, slim.fully_connected, actions_hidden_layers, scope='actions')
        joint = tf.concat([states, actions], 1)
        if joint_hidden_layers:
            joint = slim.stack(joint, slim.fully_connected, joint_hidden_layers, scope='joint')
        with slim.arg_scope([slim.fully_connected], weights_regularizer=None, weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)):
            value = slim.fully_connected(joint, num_reward_dims, activation_fn=None, normalizer_fn=None, scope='q_value')
        if num_reward_dims == 1:
            value = tf.reshape(value, [-1])
        if not for_critic_loss and num_reward_dims > 1:
            value = tf.reduce_sum(value * tf.abs(orig_states[:, -num_reward_dims:]), -1)
    return value

@gin.configurable('ddpg_actor_net')
def actor_net(states, action_spec, hidden_layers=(400, 300), normalizer_fn=None, activation_fn=tf.nn.relu, zero_obs=False, images=False):
    if False:
        while True:
            i = 10
    'Creates an actor that returns actions for the given states.\n\n  Args:\n    states: (castable to tf.float32) a [batch_size, num_state_dims] tensor\n      representing a batch of states.\n    action_spec: (BoundedTensorSpec) A tensor spec indicating the shape\n      and range of actions.\n    hidden_layers: tuple of hidden layers units.\n    normalizer_fn: Normalizer function, i.e. slim.layer_norm,\n    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...\n  Returns:\n    A tf.float32 [batch_size, num_action_dims] tensor of actions.\n  '
    with slim.arg_scope([slim.fully_connected], activation_fn=activation_fn, normalizer_fn=normalizer_fn, weights_initializer=slim.variance_scaling_initializer(factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):
        states = tf.to_float(states)
        orig_states = states
        if images or zero_obs:
            states *= tf.constant([0.0] * 2 + [1.0] * (states.shape[1] - 2))
        if hidden_layers:
            states = slim.stack(states, slim.fully_connected, hidden_layers, scope='states')
        with slim.arg_scope([slim.fully_connected], weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)):
            actions = slim.fully_connected(states, action_spec.shape.num_elements(), scope='actions', normalizer_fn=None, activation_fn=tf.nn.tanh)
            action_means = (action_spec.maximum + action_spec.minimum) / 2.0
            action_magnitudes = (action_spec.maximum - action_spec.minimum) / 2.0
            actions = action_means + action_magnitudes * actions
    return actions