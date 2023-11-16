"""A DDPG/NAF agent.

Implements the Deep Deterministic Policy Gradient (DDPG) algorithm from
"Continuous control with deep reinforcement learning" - Lilicrap et al.
https://arxiv.org/abs/1509.02971, and the Normalized Advantage Functions (NAF)
algorithm "Continuous Deep Q-Learning with Model-based Acceleration" - Gu et al.
https://arxiv.org/pdf/1603.00748.
"""
import tensorflow as tf
slim = tf.contrib.slim
import gin.tf
from utils import utils
from agents import ddpg_networks as networks

@gin.configurable
class DdpgAgent(object):
    """An RL agent that learns using the DDPG algorithm.

  Example usage:

  def critic_net(states, actions):
    ...
  def actor_net(states, num_action_dims):
    ...

  Given a tensorflow environment tf_env,
  (of type learning.deepmind.rl.environments.tensorflow.python.tfpyenvironment)

  obs_spec = tf_env.observation_spec()
  action_spec = tf_env.action_spec()

  ddpg_agent = agent.DdpgAgent(obs_spec,
                               action_spec,
                               actor_net=actor_net,
                               critic_net=critic_net)

  we can perform actions on the environment as follows:

  state = tf_env.observations()[0]
  action = ddpg_agent.actor_net(tf.expand_dims(state, 0))[0, :]
  transition_type, reward, discount = tf_env.step([action])

  Train:

  critic_loss = ddpg_agent.critic_loss(states, actions, rewards, discounts,
                                       next_states)
  actor_loss = ddpg_agent.actor_loss(states)

  critic_train_op = slim.learning.create_train_op(
      critic_loss,
      critic_optimizer,
      variables_to_train=ddpg_agent.get_trainable_critic_vars(),
  )

  actor_train_op = slim.learning.create_train_op(
      actor_loss,
      actor_optimizer,
      variables_to_train=ddpg_agent.get_trainable_actor_vars(),
  )
  """
    ACTOR_NET_SCOPE = 'actor_net'
    CRITIC_NET_SCOPE = 'critic_net'
    TARGET_ACTOR_NET_SCOPE = 'target_actor_net'
    TARGET_CRITIC_NET_SCOPE = 'target_critic_net'

    def __init__(self, observation_spec, action_spec, actor_net=networks.actor_net, critic_net=networks.critic_net, td_errors_loss=tf.losses.huber_loss, dqda_clipping=0.0, actions_regularizer=0.0, target_q_clipping=None, residual_phi=0.0, debug_summaries=False):
        if False:
            while True:
                i = 10
        "Constructs a DDPG agent.\n\n    Args:\n      observation_spec: A TensorSpec defining the observations.\n      action_spec: A BoundedTensorSpec defining the actions.\n      actor_net: A callable that creates the actor network. Must take the\n        following arguments: states, num_actions. Please see networks.actor_net\n        for an example.\n      critic_net: A callable that creates the critic network. Must take the\n        following arguments: states, actions. Please see networks.critic_net\n        for an example.\n      td_errors_loss: A callable defining the loss function for the critic\n        td error.\n      dqda_clipping: (float) clips the gradient dqda element-wise between\n        [-dqda_clipping, dqda_clipping]. Does not perform clipping if\n        dqda_clipping == 0.\n      actions_regularizer: A scalar, when positive penalizes the norm of the\n        actions. This can prevent saturation of actions for the actor_loss.\n      target_q_clipping: (tuple of floats) clips target q values within\n        (low, high) values when computing the critic loss.\n      residual_phi: (float) [0.0, 1.0] Residual algorithm parameter that\n        interpolates between Q-learning and residual gradient algorithm.\n        http://www.leemon.com/papers/1995b.pdf\n      debug_summaries: If True, add summaries to help debug behavior.\n    Raises:\n      ValueError: If 'dqda_clipping' is < 0.\n    "
        self._observation_spec = observation_spec[0]
        self._action_spec = action_spec[0]
        self._state_shape = tf.TensorShape([None]).concatenate(self._observation_spec.shape)
        self._action_shape = tf.TensorShape([None]).concatenate(self._action_spec.shape)
        self._num_action_dims = self._action_spec.shape.num_elements()
        self._scope = tf.get_variable_scope().name
        self._actor_net = tf.make_template(self.ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
        self._critic_net = tf.make_template(self.CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._target_actor_net = tf.make_template(self.TARGET_ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
        self._target_critic_net = tf.make_template(self.TARGET_CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._td_errors_loss = td_errors_loss
        if dqda_clipping < 0:
            raise ValueError('dqda_clipping must be >= 0.')
        self._dqda_clipping = dqda_clipping
        self._actions_regularizer = actions_regularizer
        self._target_q_clipping = target_q_clipping
        self._residual_phi = residual_phi
        self._debug_summaries = debug_summaries

    def _batch_state(self, state):
        if False:
            while True:
                i = 10
        'Convert state to a batched state.\n\n    Args:\n      state: Either a list/tuple with an state tensor [num_state_dims].\n    Returns:\n      A tensor [1, num_state_dims]\n    '
        if isinstance(state, (tuple, list)):
            state = state[0]
        if state.get_shape().ndims == 1:
            state = tf.expand_dims(state, 0)
        return state

    def action(self, state):
        if False:
            i = 10
            return i + 15
        'Returns the next action for the state.\n\n    Args:\n      state: A [num_state_dims] tensor representing a state.\n    Returns:\n      A [num_action_dims] tensor representing the action.\n    '
        return self.actor_net(self._batch_state(state), stop_gradients=True)[0, :]

    @gin.configurable('ddpg_sample_action')
    def sample_action(self, state, stddev=1.0):
        if False:
            print('Hello World!')
        'Returns the action for the state with additive noise.\n\n    Args:\n      state: A [num_state_dims] tensor representing a state.\n      stddev: stddev for the Ornstein-Uhlenbeck noise.\n    Returns:\n      A [num_action_dims] action tensor.\n    '
        agent_action = self.action(state)
        agent_action += tf.random_normal(tf.shape(agent_action)) * stddev
        return utils.clip_to_spec(agent_action, self._action_spec)

    def actor_net(self, states, stop_gradients=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns the output of the actor network.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      stop_gradients: (boolean) if true, gradients cannot be propogated through\n        this operation.\n    Returns:\n      A [batch_size, num_action_dims] tensor of actions.\n    Raises:\n      ValueError: If `states` does not have the expected dimensions.\n    '
        self._validate_states(states)
        actions = self._actor_net(states, self._action_spec)
        if stop_gradients:
            actions = tf.stop_gradient(actions)
        return actions

    def critic_net(self, states, actions, for_critic_loss=False):
        if False:
            for i in range(10):
                print('nop')
        "Returns the output of the critic network.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      actions: A [batch_size, num_action_dims] tensor representing a batch\n        of actions.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    Raises:\n      ValueError: If `states` or `actions' do not have the expected dimensions.\n    "
        self._validate_states(states)
        self._validate_actions(actions)
        return self._critic_net(states, actions, for_critic_loss=for_critic_loss)

    def target_actor_net(self, states):
        if False:
            print('Hello World!')
        'Returns the output of the target actor network.\n\n    The target network is used to compute stable targets for training.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      A [batch_size, num_action_dims] tensor of actions.\n    Raises:\n      ValueError: If `states` does not have the expected dimensions.\n    '
        self._validate_states(states)
        actions = self._target_actor_net(states, self._action_spec)
        return tf.stop_gradient(actions)

    def target_critic_net(self, states, actions, for_critic_loss=False):
        if False:
            while True:
                i = 10
        "Returns the output of the target critic network.\n\n    The target network is used to compute stable targets for training.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      actions: A [batch_size, num_action_dims] tensor representing a batch\n        of actions.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    Raises:\n      ValueError: If `states` or `actions' do not have the expected dimensions.\n    "
        self._validate_states(states)
        self._validate_actions(actions)
        return tf.stop_gradient(self._target_critic_net(states, actions, for_critic_loss=for_critic_loss))

    def value_net(self, states, for_critic_loss=False):
        if False:
            print('Hello World!')
        'Returns the output of the critic evaluated with the actor.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    '
        actions = self.actor_net(states)
        return self.critic_net(states, actions, for_critic_loss=for_critic_loss)

    def target_value_net(self, states, for_critic_loss=False):
        if False:
            print('Hello World!')
        'Returns the output of the target critic evaluated with the target actor.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    '
        target_actions = self.target_actor_net(states)
        return self.target_critic_net(states, target_actions, for_critic_loss=for_critic_loss)

    def critic_loss(self, states, actions, rewards, discounts, next_states):
        if False:
            while True:
                i = 10
        'Computes a loss for training the critic network.\n\n    The loss is the mean squared error between the Q value predictions of the\n    critic and Q values estimated using TD-lambda.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      actions: A [batch_size, num_action_dims] tensor representing a batch\n        of actions.\n      rewards: A [batch_size, ...] tensor representing a batch of rewards,\n        broadcastable to the critic net output.\n      discounts: A [batch_size, ...] tensor representing a batch of discounts,\n        broadcastable to the critic net output.\n      next_states: A [batch_size, num_state_dims] tensor representing a batch\n        of next states.\n    Returns:\n      A rank-0 tensor representing the critic loss.\n    Raises:\n      ValueError: If any of the inputs do not have the expected dimensions, or\n        if their batch_sizes do not match.\n    '
        self._validate_states(states)
        self._validate_actions(actions)
        self._validate_states(next_states)
        target_q_values = self.target_value_net(next_states, for_critic_loss=True)
        td_targets = target_q_values * discounts + rewards
        if self._target_q_clipping is not None:
            td_targets = tf.clip_by_value(td_targets, self._target_q_clipping[0], self._target_q_clipping[1])
        q_values = self.critic_net(states, actions, for_critic_loss=True)
        td_errors = td_targets - q_values
        if self._debug_summaries:
            gen_debug_td_error_summaries(target_q_values, q_values, td_targets, td_errors)
        loss = self._td_errors_loss(td_targets, q_values)
        if self._residual_phi > 0.0:
            residual_q_values = self.value_net(next_states, for_critic_loss=True)
            residual_td_targets = residual_q_values * discounts + rewards
            if self._target_q_clipping is not None:
                residual_td_targets = tf.clip_by_value(residual_td_targets, self._target_q_clipping[0], self._target_q_clipping[1])
            residual_td_errors = residual_td_targets - q_values
            residual_loss = self._td_errors_loss(residual_td_targets, residual_q_values)
            loss = loss * (1.0 - self._residual_phi) + residual_loss * self._residual_phi
        return loss

    def actor_loss(self, states):
        if False:
            return 10
        'Computes a loss for training the actor network.\n\n    Note that output does not represent an actual loss. It is called a loss only\n    in the sense that its gradient w.r.t. the actor network weights is the\n    correct gradient for training the actor network,\n    i.e. dloss/dweights = (dq/da)*(da/dweights)\n    which is the gradient used in Algorithm 1 of Lilicrap et al.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      A rank-0 tensor representing the actor loss.\n    Raises:\n      ValueError: If `states` does not have the expected dimensions.\n    '
        self._validate_states(states)
        actions = self.actor_net(states, stop_gradients=False)
        critic_values = self.critic_net(states, actions)
        q_values = self.critic_function(critic_values, states)
        dqda = tf.gradients([q_values], [actions])[0]
        dqda_unclipped = dqda
        if self._dqda_clipping > 0:
            dqda = tf.clip_by_value(dqda, -self._dqda_clipping, self._dqda_clipping)
        actions_norm = tf.norm(actions)
        if self._debug_summaries:
            with tf.name_scope('dqda'):
                tf.summary.scalar('actions_norm', actions_norm)
                tf.summary.histogram('dqda', dqda)
                tf.summary.histogram('dqda_unclipped', dqda_unclipped)
                tf.summary.histogram('actions', actions)
                for a in range(self._num_action_dims):
                    tf.summary.histogram('dqda_unclipped_%d' % a, dqda_unclipped[:, a])
                    tf.summary.histogram('dqda_%d' % a, dqda[:, a])
        actions_norm *= self._actions_regularizer
        return slim.losses.mean_squared_error(tf.stop_gradient(dqda + actions), actions, scope='actor_loss') + actions_norm

    @gin.configurable('ddpg_critic_function')
    def critic_function(self, critic_values, states, weights=None):
        if False:
            print('Hello World!')
        'Computes q values based on critic_net outputs, states, and weights.\n\n    Args:\n      critic_values: A tf.float32 [batch_size, ...] tensor representing outputs\n        from the critic net.\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      weights: A list or Numpy array or tensor with a shape broadcastable to\n        `critic_values`.\n    Returns:\n      A tf.float32 [batch_size] tensor representing q values.\n    '
        del states
        if weights is not None:
            weights = tf.convert_to_tensor(weights, dtype=critic_values.dtype)
            critic_values *= weights
        if critic_values.shape.ndims > 1:
            critic_values = tf.reduce_sum(critic_values, range(1, critic_values.shape.ndims))
        critic_values.shape.assert_has_rank(1)
        return critic_values

    @gin.configurable('ddpg_update_targets')
    def update_targets(self, tau=1.0):
        if False:
            return 10
        'Performs a soft update of the target network parameters.\n\n    For each weight w_s in the actor/critic networks, and its corresponding\n    weight w_t in the target actor/critic networks, a soft update is:\n    w_t = (1- tau) x w_t + tau x ws\n\n    Args:\n      tau: A float scalar in [0, 1]\n    Returns:\n      An operation that performs a soft update of the target network parameters.\n    Raises:\n      ValueError: If `tau` is not in [0, 1].\n    '
        if tau < 0 or tau > 1:
            raise ValueError('Input `tau` should be in [0, 1].')
        update_actor = utils.soft_variables_update(slim.get_trainable_variables(utils.join_scope(self._scope, self.ACTOR_NET_SCOPE)), slim.get_trainable_variables(utils.join_scope(self._scope, self.TARGET_ACTOR_NET_SCOPE)), tau)
        update_critic = utils.soft_variables_update(slim.get_trainable_variables(utils.join_scope(self._scope, self.CRITIC_NET_SCOPE)), slim.get_trainable_variables(utils.join_scope(self._scope, self.TARGET_CRITIC_NET_SCOPE)), tau)
        return tf.group(update_actor, update_critic, name='update_targets')

    def get_trainable_critic_vars(self):
        if False:
            return 10
        'Returns a list of trainable variables in the critic network.\n\n    Returns:\n      A list of trainable variables in the critic network.\n    '
        return slim.get_trainable_variables(utils.join_scope(self._scope, self.CRITIC_NET_SCOPE))

    def get_trainable_actor_vars(self):
        if False:
            while True:
                i = 10
        'Returns a list of trainable variables in the actor network.\n\n    Returns:\n      A list of trainable variables in the actor network.\n    '
        return slim.get_trainable_variables(utils.join_scope(self._scope, self.ACTOR_NET_SCOPE))

    def get_critic_vars(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of all variables in the critic network.\n\n    Returns:\n      A list of trainable variables in the critic network.\n    '
        return slim.get_model_variables(utils.join_scope(self._scope, self.CRITIC_NET_SCOPE))

    def get_actor_vars(self):
        if False:
            while True:
                i = 10
        'Returns a list of all variables in the actor network.\n\n    Returns:\n      A list of trainable variables in the actor network.\n    '
        return slim.get_model_variables(utils.join_scope(self._scope, self.ACTOR_NET_SCOPE))

    def _validate_states(self, states):
        if False:
            for i in range(10):
                print('nop')
        'Raises a value error if `states` does not have the expected shape.\n\n    Args:\n      states: A tensor.\n    Raises:\n      ValueError: If states.shape or states.dtype are not compatible with\n        observation_spec.\n    '
        states.shape.assert_is_compatible_with(self._state_shape)
        if not states.dtype.is_compatible_with(self._observation_spec.dtype):
            raise ValueError('states.dtype={} is not compatible with observation_spec.dtype={}'.format(states.dtype, self._observation_spec.dtype))

    def _validate_actions(self, actions):
        if False:
            return 10
        'Raises a value error if `actions` does not have the expected shape.\n\n    Args:\n      actions: A tensor.\n    Raises:\n      ValueError: If actions.shape or actions.dtype are not compatible with\n        action_spec.\n    '
        actions.shape.assert_is_compatible_with(self._action_shape)
        if not actions.dtype.is_compatible_with(self._action_spec.dtype):
            raise ValueError('actions.dtype={} is not compatible with action_spec.dtype={}'.format(actions.dtype, self._action_spec.dtype))

@gin.configurable
class TD3Agent(DdpgAgent):
    """An RL agent that learns using the TD3 algorithm."""
    ACTOR_NET_SCOPE = 'actor_net'
    CRITIC_NET_SCOPE = 'critic_net'
    CRITIC_NET2_SCOPE = 'critic_net2'
    TARGET_ACTOR_NET_SCOPE = 'target_actor_net'
    TARGET_CRITIC_NET_SCOPE = 'target_critic_net'
    TARGET_CRITIC_NET2_SCOPE = 'target_critic_net2'

    def __init__(self, observation_spec, action_spec, actor_net=networks.actor_net, critic_net=networks.critic_net, td_errors_loss=tf.losses.huber_loss, dqda_clipping=0.0, actions_regularizer=0.0, target_q_clipping=None, residual_phi=0.0, debug_summaries=False):
        if False:
            i = 10
            return i + 15
        "Constructs a TD3 agent.\n\n    Args:\n      observation_spec: A TensorSpec defining the observations.\n      action_spec: A BoundedTensorSpec defining the actions.\n      actor_net: A callable that creates the actor network. Must take the\n        following arguments: states, num_actions. Please see networks.actor_net\n        for an example.\n      critic_net: A callable that creates the critic network. Must take the\n        following arguments: states, actions. Please see networks.critic_net\n        for an example.\n      td_errors_loss: A callable defining the loss function for the critic\n        td error.\n      dqda_clipping: (float) clips the gradient dqda element-wise between\n        [-dqda_clipping, dqda_clipping]. Does not perform clipping if\n        dqda_clipping == 0.\n      actions_regularizer: A scalar, when positive penalizes the norm of the\n        actions. This can prevent saturation of actions for the actor_loss.\n      target_q_clipping: (tuple of floats) clips target q values within\n        (low, high) values when computing the critic loss.\n      residual_phi: (float) [0.0, 1.0] Residual algorithm parameter that\n        interpolates between Q-learning and residual gradient algorithm.\n        http://www.leemon.com/papers/1995b.pdf\n      debug_summaries: If True, add summaries to help debug behavior.\n    Raises:\n      ValueError: If 'dqda_clipping' is < 0.\n    "
        self._observation_spec = observation_spec[0]
        self._action_spec = action_spec[0]
        self._state_shape = tf.TensorShape([None]).concatenate(self._observation_spec.shape)
        self._action_shape = tf.TensorShape([None]).concatenate(self._action_spec.shape)
        self._num_action_dims = self._action_spec.shape.num_elements()
        self._scope = tf.get_variable_scope().name
        self._actor_net = tf.make_template(self.ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
        self._critic_net = tf.make_template(self.CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._critic_net2 = tf.make_template(self.CRITIC_NET2_SCOPE, critic_net, create_scope_now_=True)
        self._target_actor_net = tf.make_template(self.TARGET_ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
        self._target_critic_net = tf.make_template(self.TARGET_CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._target_critic_net2 = tf.make_template(self.TARGET_CRITIC_NET2_SCOPE, critic_net, create_scope_now_=True)
        self._td_errors_loss = td_errors_loss
        if dqda_clipping < 0:
            raise ValueError('dqda_clipping must be >= 0.')
        self._dqda_clipping = dqda_clipping
        self._actions_regularizer = actions_regularizer
        self._target_q_clipping = target_q_clipping
        self._residual_phi = residual_phi
        self._debug_summaries = debug_summaries

    def get_trainable_critic_vars(self):
        if False:
            while True:
                i = 10
        'Returns a list of trainable variables in the critic network.\n    NOTE: This gets the vars of both critic networks.\n\n    Returns:\n      A list of trainable variables in the critic network.\n    '
        return slim.get_trainable_variables(utils.join_scope(self._scope, self.CRITIC_NET_SCOPE))

    def critic_net(self, states, actions, for_critic_loss=False):
        if False:
            while True:
                i = 10
        "Returns the output of the critic network.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      actions: A [batch_size, num_action_dims] tensor representing a batch\n        of actions.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    Raises:\n      ValueError: If `states` or `actions' do not have the expected dimensions.\n    "
        values1 = self._critic_net(states, actions, for_critic_loss=for_critic_loss)
        values2 = self._critic_net2(states, actions, for_critic_loss=for_critic_loss)
        if for_critic_loss:
            return (values1, values2)
        return values1

    def target_critic_net(self, states, actions, for_critic_loss=False):
        if False:
            return 10
        "Returns the output of the target critic network.\n\n    The target network is used to compute stable targets for training.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n      actions: A [batch_size, num_action_dims] tensor representing a batch\n        of actions.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    Raises:\n      ValueError: If `states` or `actions' do not have the expected dimensions.\n    "
        self._validate_states(states)
        self._validate_actions(actions)
        values1 = tf.stop_gradient(self._target_critic_net(states, actions, for_critic_loss=for_critic_loss))
        values2 = tf.stop_gradient(self._target_critic_net2(states, actions, for_critic_loss=for_critic_loss))
        if for_critic_loss:
            return (values1, values2)
        return values1

    def value_net(self, states, for_critic_loss=False):
        if False:
            return 10
        'Returns the output of the critic evaluated with the actor.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    '
        actions = self.actor_net(states)
        return self.critic_net(states, actions, for_critic_loss=for_critic_loss)

    def target_value_net(self, states, for_critic_loss=False):
        if False:
            print('Hello World!')
        'Returns the output of the target critic evaluated with the target actor.\n\n    Args:\n      states: A [batch_size, num_state_dims] tensor representing a batch\n        of states.\n    Returns:\n      q values: A [batch_size] tensor of q values.\n    '
        target_actions = self.target_actor_net(states)
        noise = tf.clip_by_value(tf.random_normal(tf.shape(target_actions), stddev=0.2), -0.5, 0.5)
        (values1, values2) = self.target_critic_net(states, target_actions + noise, for_critic_loss=for_critic_loss)
        values = tf.minimum(values1, values2)
        return (values, values)

    @gin.configurable('td3_update_targets')
    def update_targets(self, tau=1.0):
        if False:
            while True:
                i = 10
        'Performs a soft update of the target network parameters.\n\n    For each weight w_s in the actor/critic networks, and its corresponding\n    weight w_t in the target actor/critic networks, a soft update is:\n    w_t = (1- tau) x w_t + tau x ws\n\n    Args:\n      tau: A float scalar in [0, 1]\n    Returns:\n      An operation that performs a soft update of the target network parameters.\n    Raises:\n      ValueError: If `tau` is not in [0, 1].\n    '
        if tau < 0 or tau > 1:
            raise ValueError('Input `tau` should be in [0, 1].')
        update_actor = utils.soft_variables_update(slim.get_trainable_variables(utils.join_scope(self._scope, self.ACTOR_NET_SCOPE)), slim.get_trainable_variables(utils.join_scope(self._scope, self.TARGET_ACTOR_NET_SCOPE)), tau)
        update_critic = utils.soft_variables_update(slim.get_trainable_variables(utils.join_scope(self._scope, self.CRITIC_NET_SCOPE)), slim.get_trainable_variables(utils.join_scope(self._scope, self.TARGET_CRITIC_NET_SCOPE)), tau)
        return tf.group(update_actor, update_critic, name='update_targets')

def gen_debug_td_error_summaries(target_q_values, q_values, td_targets, td_errors):
    if False:
        print('Hello World!')
    'Generates debug summaries for critic given a set of batch samples.\n\n  Args:\n    target_q_values: set of predicted next stage values.\n    q_values: current predicted value for the critic network.\n    td_targets: discounted target_q_values with added next stage reward.\n    td_errors: the different between td_targets and q_values.\n  '
    with tf.name_scope('td_errors'):
        tf.summary.histogram('td_targets', td_targets)
        tf.summary.histogram('q_values', q_values)
        tf.summary.histogram('target_q_values', target_q_values)
        tf.summary.histogram('td_errors', td_errors)
        with tf.name_scope('td_targets'):
            tf.summary.scalar('mean', tf.reduce_mean(td_targets))
            tf.summary.scalar('max', tf.reduce_max(td_targets))
            tf.summary.scalar('min', tf.reduce_min(td_targets))
        with tf.name_scope('q_values'):
            tf.summary.scalar('mean', tf.reduce_mean(q_values))
            tf.summary.scalar('max', tf.reduce_max(q_values))
            tf.summary.scalar('min', tf.reduce_min(q_values))
        with tf.name_scope('target_q_values'):
            tf.summary.scalar('mean', tf.reduce_mean(target_q_values))
            tf.summary.scalar('max', tf.reduce_max(target_q_values))
            tf.summary.scalar('min', tf.reduce_min(target_q_values))
        with tf.name_scope('td_errors'):
            tf.summary.scalar('mean', tf.reduce_mean(td_errors))
            tf.summary.scalar('max', tf.reduce_max(td_errors))
            tf.summary.scalar('min', tf.reduce_min(td_errors))
            tf.summary.scalar('mean_abs', tf.reduce_mean(tf.abs(td_errors)))