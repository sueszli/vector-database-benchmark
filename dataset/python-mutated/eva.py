"""Implements an Ephemeral Value Adjustment Agent.

See https://arxiv.org/abs/1810.08163.

The algorithm queries trajectories from a replay buffer based on similarities
to embedding representations and uses a parametric model to compute values for
counterfactual state-action pairs when integrating across those trajectories.
Finally, a weighted average between the parametric (DQN in this case) and the
non-parametric model is used to compute the policy.
"""
import collections
import copy
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.algorithms import dqn
tf.disable_v2_behavior()
MEM_KEY_NAME = 'embedding'
ValueBufferElement = collections.namedtuple('ValueElement', 'embedding value')
ReplayBufferElement = collections.namedtuple('ReplayElement', 'embedding info_state action reward next_info_state is_final_step legal_actions_mask')

class QueryableFixedSizeRingBuffer(dqn.ReplayBuffer):
    """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.  This extends the DQN replay
  buffer by allowing the contents to be fetched by L2 proximity to a query
  value.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

    def knn(self, key, key_name, k, trajectory_len=1):
        if False:
            print('Hello World!')
        'Computes top-k neighbours based on L2 distance.\n\n    Args:\n      key: (np.array) key value to query memory.\n      key_name:  (str) attribute name of key in memory elements.\n      k: (int) number of neighbours to fetch.\n      trajectory_len: (int) length of trajectory to fetch from replay buffer.\n\n    Returns:\n      List of tuples (L2 negative distance, BufferElement) sorted in increasing\n      order by the negative L2 distqances  from the key.\n    '
        distances = [(np.linalg.norm(getattr(sample, key_name) - key, 2, axis=0), sample) for sample in self._data]
        return sorted(distances, key=lambda v: -v[0])[:k]

class EVAAgent(object):
    """Implements a solver for Ephemeral VAlue Adjustment.

  See https://arxiv.org/abs/1810.08163.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.
  """

    def __init__(self, session, game, player_id, state_size, num_actions, embedding_network_layers=(128,), embedding_size=16, dqn_hidden_layers=(128, 128), batch_size=16, trajectory_len=10, num_neighbours=5, learning_rate=0.0001, mixing_parameter=0.9, memory_capacity=int(1000000.0), discount_factor=1.0, update_target_network_every=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_duration=int(10000.0), embedding_as_parametric_input=False):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the Ephemeral VAlue Adjustment algorithm.\n\n    Args:\n      session: (tf.Session) TensorFlow session.\n      game: (rl_environment.Environment) Open Spiel game.\n      player_id: (int) Player id for this player.\n      state_size: (int) Size of info state vector.\n      num_actions: (int) number of actions.\n      embedding_network_layers: (list[int]) Layer sizes of strategy net MLP.\n      embedding_size: (int) Size of memory embeddings.\n      dqn_hidden_layers: (list(int)) MLP layer sizes of DQN network.\n      batch_size: (int) Size of batches for DQN learning steps.\n      trajectory_len: (int) Length of trajectories from replay buffer.\n      num_neighbours: (int) Number of neighbours to fetch from replay buffer.\n      learning_rate: (float) Learning rate.\n      mixing_parameter: (float) Value mixing parameter between 0 and 1.\n      memory_capacity: Number af samples that can be stored in memory.\n      discount_factor: (float) Discount factor for Q-Learning.\n      update_target_network_every: How often to update DQN target network.\n      epsilon_start: (float) Starting epsilon-greedy value.\n      epsilon_end: (float) Final epsilon-greedy value.\n      epsilon_decay_duration: (float) Number of steps over which epsilon decays.\n      embedding_as_parametric_input: (bool) Whether we use embeddings as input\n        to the parametric model.\n    '
        assert mixing_parameter >= 0 and mixing_parameter <= 1
        self._game = game
        self._session = session
        self.player_id = player_id
        self._env = game
        self._num_actions = num_actions
        self._info_state_size = state_size
        self._embedding_size = embedding_size
        self._lambda = mixing_parameter
        self._trajectory_len = trajectory_len
        self._num_neighbours = num_neighbours
        self._discount = discount_factor
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration
        self._last_time_step = None
        self._last_action = None
        self._embedding_as_parametric_input = embedding_as_parametric_input
        self._info_state_ph = tf.placeholder(shape=[None, self._info_state_size], dtype=tf.float32, name='info_state_ph')
        self._embedding_network = simple_nets.MLP(self._info_state_size, list(embedding_network_layers), embedding_size)
        self._embedding = self._embedding_network(self._info_state_ph)
        if not isinstance(memory_capacity, int):
            raise ValueError('Memory capacity not an integer.')
        self._agent = dqn.DQN(session, player_id, state_representation_size=self._info_state_size, num_actions=self._num_actions, hidden_layers_sizes=list(dqn_hidden_layers), replay_buffer_capacity=memory_capacity, replay_buffer_class=QueryableFixedSizeRingBuffer, batch_size=batch_size, learning_rate=learning_rate, update_target_network_every=update_target_network_every, learn_every=batch_size, discount_factor=1.0, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_duration=int(1000000.0))
        self._value_buffer = QueryableFixedSizeRingBuffer(memory_capacity)
        self._replay_buffer = self._agent.replay_buffer
        self._v_np = collections.defaultdict(float)
        self._q_np = collections.defaultdict(lambda : [0] * self._num_actions)
        self._q_eva = collections.defaultdict(lambda : [0] * self._num_actions)

    @property
    def env(self):
        if False:
            for i in range(10):
                print('nop')
        return self._env

    @property
    def loss(self):
        if False:
            print('Hello World!')
        return self._agent.loss

    def _add_transition_value(self, infostate_embedding, value):
        if False:
            print('Hello World!')
        'Adds the embedding and value to the ValueBuffer.\n\n    Args:\n      infostate_embedding: (np.array) embeddig vector.\n      value: (float) Value associated with state embeding.\n    '
        transition = ValueBufferElement(embedding=infostate_embedding, value=value)
        self._value_buffer.add(transition)

    def _add_transition_replay(self, infostate_embedding, time_step):
        if False:
            i = 10
            return i + 15
        'Adds the new transition using `time_step` to the replay buffer.\n\n    Adds the transition from `self._prev_timestep` to `time_step` by\n    `self._prev_action`.\n\n    Args:\n      infostate_embedding: embeddig vector.\n      time_step: an instance of rl_environment.TimeStep.\n    '
        prev_timestep = self._last_time_step
        assert prev_timestep is not None
        legal_actions = prev_timestep.observations['legal_actions'][self.player_id]
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        reward = time_step.rewards[self.player_id] if time_step.rewards else 0.0
        transition = ReplayBufferElement(embedding=infostate_embedding, info_state=prev_timestep.observations['info_state'][self.player_id], action=self._last_action, reward=reward, next_info_state=time_step.observations['info_state'][self.player_id], is_final_step=float(time_step.last()), legal_actions_mask=legal_actions_mask)
        self._replay_buffer.add(transition)

    def step(self, time_step, is_evaluation=False):
        if False:
            i = 10
            return i + 15
        'Returns the action to be taken and updates the value functions.\n\n    Args:\n      time_step: an instance of rl_environment.TimeStep.\n      is_evaluation: bool, whether this is a training or evaluation call.\n\n    Returns:\n      A `rl_agent.StepOutput` containing the action probs and chosen action.\n    '
        if not time_step.last():
            info_state = time_step.observations['info_state'][self.player_id]
            legal_actions = time_step.observations['legal_actions'][self.player_id]
            epsilon = self._get_epsilon(self._agent.step_counter, is_evaluation)
            (action, probs) = self._epsilon_greedy(self._q_eva[tuple(info_state)], legal_actions, epsilon)
        if not is_evaluation and self._last_time_step is not None:
            info_state = self._last_time_step.observations['info_state'][self.player_id]
            legal_actions = self._last_time_step.observations['legal_actions'][self.player_id]
            epsilon = self._get_epsilon(self._agent.step_counter, is_evaluation)
            infostate_embedding = self._session.run(self._embedding, feed_dict={self._info_state_ph: np.expand_dims(info_state, axis=0)})[0]
            neighbours_value = self._value_buffer.knn(infostate_embedding, MEM_KEY_NAME, self._num_neighbours, 1)
            neighbours_replay = self._replay_buffer.knn(infostate_embedding, MEM_KEY_NAME, self._num_neighbours, self._trajectory_len)
            if self._embedding_as_parametric_input:
                last_time_step_copy = copy.deepcopy(self._last_time_step)
                last_time_step_copy.observations['info_state'][self.player_id] = infostate_embedding
                self._agent.step(last_time_step_copy, add_transition_record=False)
            else:
                self._agent.step(self._last_time_step, add_transition_record=False)
            q_values = self._session.run(self._agent.q_values, feed_dict={self._agent.info_state_ph: np.expand_dims(info_state, axis=0)})[0]
            for a in legal_actions:
                q_theta = q_values[a]
                self._q_eva[tuple(info_state)][a] = self._lambda * q_theta + (1 - self._lambda) * sum([elem[1].value for elem in neighbours_value]) / self._num_neighbours
            self._add_transition_replay(infostate_embedding, time_step)
            self._trajectory_centric_planning(neighbours_replay)
            self._add_transition_value(infostate_embedding, self._q_np[tuple(info_state)][self._last_action])
        if time_step.last():
            self._last_time_step = None
            self._last_action = None
            return
        self._last_time_step = time_step
        self._last_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    def _trajectory_centric_planning(self, trajectories):
        if False:
            print('Hello World!')
        'Performs trajectory centric planning.\n\n    Uses trajectories from the replay buffer to update the non-parametric values\n    while supplying counter-factual values with the parametric model.\n\n    Args:\n      trajectories: Current OpenSpiel game state.\n    '
        for t in range(len(trajectories) - 1, 0, -1):
            elem = trajectories[t][1]
            s_tp1 = tuple(elem.next_info_state)
            s_t = tuple(elem.info_state)
            a_t = elem.action
            r_t = elem.reward
            legal_actions = elem.legal_actions_mask
            if t < len(trajectories) - 1:
                for action in range(len(legal_actions)):
                    if not legal_actions[action]:
                        continue
                    if action == elem.action:
                        self._q_np[s_t][a_t] = r_t + self._discount * self._v_np[s_tp1]
                    else:
                        q_values_parametric = self._session.run(self._agent.q_values, feed_dict={self._agent.info_state_ph: np.expand_dims(elem.info_state, axis=0)})
                        self._q_np[s_t][a_t] = q_values_parametric[0][action]
            if t == len(trajectories) - 1:
                q_values_parametric = self._session.run(self._agent.q_values, feed_dict={self._agent.info_state_ph: np.expand_dims(elem.info_state, axis=0)})
                self._v_np[s_t] = np.max(q_values_parametric)
            else:
                self._v_np[s_t] = max(self._q_np[s_t])

    def _epsilon_greedy(self, q_values, legal_actions, epsilon):
        if False:
            while True:
                i = 10
        'Returns a valid epsilon-greedy action and valid action probs.\n\n    Action probabilities are given by a softmax over legal q-values.\n\n    Args:\n      q_values: list of Q-values by action.\n      legal_actions: list of legal actions at `info_state`.\n      epsilon: float, probability of taking an exploratory action.\n\n    Returns:\n      A valid epsilon-greedy action and valid action probabilities.\n    '
        probs = np.zeros(self._num_actions)
        q_values = np.array(q_values)
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            legal_q_values = q_values[legal_actions]
            action = legal_actions[np.argmax(legal_q_values)]
            max_q = np.max(legal_q_values)
            e_x = np.exp(legal_q_values - max_q)
            probs[legal_actions] = e_x / e_x.sum(axis=0)
        return (action, probs)

    def _get_epsilon(self, step_counter, is_evaluation):
        if False:
            while True:
                i = 10
        'Returns the evaluation or decayed epsilon value.'
        if is_evaluation:
            return 0.0
        decay_steps = min(step_counter, self._epsilon_decay_duration)
        decayed_epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * (1 - decay_steps / self._epsilon_decay_duration)
        return decayed_epsilon

    def action_probabilities(self, state):
        if False:
            while True:
                i = 10
        'Returns action probabilites dict for a single batch.'
        if hasattr(state, 'information_state_tensor'):
            state_rep = tuple(state.information_state_tensor(self.player_id))
        elif hasattr(state, 'observation_tensor'):
            state_rep = tuple(state.observation_tensor(self.player_id))
        else:
            raise AttributeError('Unable to extract normalized state vector.')
        legal_actions = state.legal_actions(self.player_id)
        if legal_actions:
            (_, probs) = self._epsilon_greedy(self._q_eva[state_rep], legal_actions, epsilon=0.0)
            return {a: probs[a] for a in range(self._num_actions)}
        else:
            raise ValueError('Node has no legal actions to take.')