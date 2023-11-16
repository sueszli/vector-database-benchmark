"""Neural Fictitious Self-Play (NFSP) agent implemented in TensorFlow.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""
import collections
import contextlib
import enum
import os
import random
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.algorithms import dqn
tf.disable_v2_behavior()
Transition = collections.namedtuple('Transition', 'info_state action_probs legal_actions_mask')
MODE = enum.Enum('mode', 'best_response average_policy')

class NFSP(rl_agent.AbstractAgent):
    """NFSP Agent implementation in TensorFlow.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

    def __init__(self, session, player_id, state_representation_size, num_actions, hidden_layers_sizes, reservoir_buffer_capacity, anticipatory_param, batch_size=128, rl_learning_rate=0.01, sl_learning_rate=0.01, min_buffer_size_to_learn=1000, learn_every=64, optimizer_str='sgd', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the `NFSP` agent.'
        self.player_id = player_id
        self._session = session
        self._num_actions = num_actions
        self._layer_sizes = hidden_layers_sizes
        self._batch_size = batch_size
        self._learn_every = learn_every
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn
        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None
        self._step_counter = 0
        kwargs.update({'batch_size': batch_size, 'learning_rate': rl_learning_rate, 'learn_every': learn_every, 'min_buffer_size_to_learn': min_buffer_size_to_learn, 'optimizer_str': optimizer_str})
        self._rl_agent = dqn.DQN(session, player_id, state_representation_size, num_actions, hidden_layers_sizes, **kwargs)
        self._last_rl_loss_value = lambda : self._rl_agent.loss
        self._last_sl_loss_value = None
        self._info_state_ph = tf.placeholder(shape=[None, state_representation_size], dtype=tf.float32, name='info_state_ph')
        self._action_probs_ph = tf.placeholder(shape=[None, num_actions], dtype=tf.float32, name='action_probs_ph')
        self._legal_actions_mask_ph = tf.placeholder(shape=[None, num_actions], dtype=tf.float32, name='legal_actions_mask_ph')
        self._avg_network = simple_nets.MLP(state_representation_size, self._layer_sizes, num_actions)
        self._avg_policy = self._avg_network(self._info_state_ph)
        self._avg_policy_probs = tf.nn.softmax(self._avg_policy)
        self._savers = [('q_network', tf.train.Saver(self._rl_agent._q_network.variables)), ('avg_network', tf.train.Saver(self._avg_network.variables))]
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self._action_probs_ph), logits=self._avg_policy))
        if optimizer_str == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=sl_learning_rate)
        elif optimizer_str == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=sl_learning_rate)
        else:
            raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")
        self._learn_step = optimizer.minimize(self._loss)
        self._sample_episode_policy()

    @contextlib.contextmanager
    def temp_mode_as(self, mode):
        if False:
            return 10
        'Context manager to temporarily overwrite the mode.'
        previous_mode = self._mode
        self._mode = mode
        yield
        self._mode = previous_mode

    def get_step_counter(self):
        if False:
            while True:
                i = 10
        return self._step_counter

    def _sample_episode_policy(self):
        if False:
            for i in range(10):
                print('nop')
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy

    def _act(self, info_state, legal_actions):
        if False:
            while True:
                i = 10
        info_state = np.reshape(info_state, [1, -1])
        (action_values, action_probs) = self._session.run([self._avg_policy, self._avg_policy_probs], feed_dict={self._info_state_ph: info_state})
        self._last_action_values = action_values[0]
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = action_probs[0][legal_actions]
        probs /= sum(probs)
        action = np.random.choice(len(probs), p=probs)
        return (action, probs)

    @property
    def mode(self):
        if False:
            return 10
        return self._mode

    @property
    def loss(self):
        if False:
            print('Hello World!')
        return (self._last_sl_loss_value, self._last_rl_loss_value())

    def step(self, time_step, is_evaluation=False):
        if False:
            i = 10
            return i + 15
        'Returns the action to be taken and updates the Q-networks if needed.\n\n    Args:\n      time_step: an instance of rl_environment.TimeStep.\n      is_evaluation: bool, whether this is a training or evaluation call.\n\n    Returns:\n      A `rl_agent.StepOutput` containing the action probs and chosen action.\n    '
        if self._mode == MODE.best_response:
            agent_output = self._rl_agent.step(time_step, is_evaluation)
            if not is_evaluation and (not time_step.last()):
                self._add_transition(time_step, agent_output)
        elif self._mode == MODE.average_policy:
            if not time_step.last():
                info_state = time_step.observations['info_state'][self.player_id]
                legal_actions = time_step.observations['legal_actions'][self.player_id]
                (action, probs) = self._act(info_state, legal_actions)
                agent_output = rl_agent.StepOutput(action=action, probs=probs)
            if self._prev_timestep and (not is_evaluation):
                self._rl_agent.add_transition(self._prev_timestep, self._prev_action, time_step)
        else:
            raise ValueError('Invalid mode ({})'.format(self._mode))
        if not is_evaluation:
            self._step_counter += 1
            if self._step_counter % self._learn_every == 0:
                self._last_sl_loss_value = self._learn()
                if self._mode == MODE.average_policy:
                    self._rl_agent.learn()
            if time_step.last():
                self._sample_episode_policy()
                self._prev_timestep = None
                self._prev_action = None
                return
            else:
                self._prev_timestep = time_step
                self._prev_action = agent_output.action
        return agent_output

    def _add_transition(self, time_step, agent_output):
        if False:
            print('Hello World!')
        'Adds the new transition using `time_step` to the reservoir buffer.\n\n    Transitions are in the form (time_step, agent_output.probs, legal_mask).\n\n    Args:\n      time_step: an instance of rl_environment.TimeStep.\n      agent_output: an instance of rl_agent.StepOutput.\n    '
        legal_actions = time_step.observations['legal_actions'][self.player_id]
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(info_state=time_step.observations['info_state'][self.player_id][:], action_probs=agent_output.probs, legal_actions_mask=legal_actions_mask)
        self._reservoir_buffer.add(transition)

    def _learn(self):
        if False:
            print('Hello World!')
        'Compute the loss on sampled transitions and perform a avg-network update.\n\n    If there are not enough elements in the buffer, no loss is computed and\n    `None` is returned instead.\n\n    Returns:\n      The average loss obtained on this batch of transitions or `None`.\n    '
        if len(self._reservoir_buffer) < self._batch_size or len(self._reservoir_buffer) < self._min_buffer_size_to_learn:
            return None
        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_probs = [t.action_probs for t in transitions]
        legal_actions_mask = [t.legal_actions_mask for t in transitions]
        (loss, _) = self._session.run([self._loss, self._learn_step], feed_dict={self._info_state_ph: info_states, self._action_probs_ph: action_probs, self._legal_actions_mask_ph: legal_actions_mask})
        return loss

    def _full_checkpoint_name(self, checkpoint_dir, name):
        if False:
            return 10
        checkpoint_filename = '_'.join([name, 'pid' + str(self.player_id)])
        return os.path.join(checkpoint_dir, checkpoint_filename)

    def _latest_checkpoint_filename(self, name):
        if False:
            for i in range(10):
                print('nop')
        checkpoint_filename = '_'.join([name, 'pid' + str(self.player_id)])
        return checkpoint_filename + '_latest'

    def save(self, checkpoint_dir):
        if False:
            i = 10
            return i + 15
        "Saves the average policy network and the inner RL agent's q-network.\n\n    Note that this does not save the experience replay buffers and should\n    only be used to restore the agent's policy, not resume training.\n\n    Args:\n      checkpoint_dir: directory where checkpoints will be saved.\n    "
        for (name, saver) in self._savers:
            path = saver.save(self._session, self._full_checkpoint_name(checkpoint_dir, name), latest_filename=self._latest_checkpoint_filename(name))
            logging.info('Saved to path: %s', path)

    def has_checkpoint(self, checkpoint_dir):
        if False:
            print('Hello World!')
        for (name, _) in self._savers:
            if tf.train.latest_checkpoint(self._full_checkpoint_name(checkpoint_dir, name), os.path.join(checkpoint_dir, self._latest_checkpoint_filename(name))) is None:
                return False
        return True

    def restore(self, checkpoint_dir):
        if False:
            return 10
        "Restores the average policy network and the inner RL agent's q-network.\n\n    Note that this does not restore the experience replay buffers and should\n    only be used to restore the agent's policy, not resume training.\n\n    Args:\n      checkpoint_dir: directory from which checkpoints will be restored.\n    "
        for (name, saver) in self._savers:
            full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
            logging.info('Restoring checkpoint: %s', full_checkpoint_dir)
            saver.restore(self._session, full_checkpoint_dir)

class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

    def __init__(self, reservoir_buffer_capacity):
        if False:
            for i in range(10):
                print('nop')
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        if False:
            while True:
                i = 10
        'Potentially adds `element` to the reservoir buffer.\n\n    Args:\n      element: data to be added to the reservoir buffer.\n    '
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        if False:
            while True:
                i = 10
        'Returns `num_samples` uniformly sampled from the buffer.\n\n    Args:\n      num_samples: `int`, number of samples to draw.\n\n    Returns:\n      An iterable over `num_samples` random elements of the buffer.\n\n    Raises:\n      ValueError: If there are less than `num_samples` elements in the buffer\n    '
        if len(self._data) < num_samples:
            raise ValueError('{} elements could not be sampled from size {}'.format(num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        if False:
            print('Hello World!')
        self._data = []
        self._add_calls = 0

    def __len__(self):
        if False:
            return 10
        return len(self._data)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._data)