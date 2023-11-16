from argparse import ArgumentParser
import gym
import numpy as np
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_parameter_schedule, momentum_schedule
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """

    def __init__(self, size, sample_shape, history_length=4):
        if False:
            for i in range(10):
                print('nop')
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the number of items currently present in the memory\n        Returns: Int >= 0\n        '
        return self._count

    def append(self, state, action, reward, done):
        if False:
            return 10
        ' Appends the specified transition to the memory.\n\n        Attributes:\n            state (Tensor[sample_shape]): The state to append\n            action (int): An integer representing the action done\n            reward (float): An integer representing the reward received for doing this action\n            done (bool): A boolean specifying if this state is a terminal (episode has finished)\n        '
        assert state.shape == self._state_shape, 'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done
        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        if False:
            print('Hello World!')
        ' Generate size random integers mapping indices in the memory.\n            The returned indices can be retrieved using #get_state().\n            See the method #minibatch() if you want to retrieve samples directly.\n\n        Attributes:\n            size (int): The minibatch size\n\n        Returns:\n             Indexes of the sampled states ([int])\n        '
        (count, pos, history_len, terminals) = (self._count - 1, self._pos, self._history_length, self._terminals)
        indexes = []
        while len(indexes) < size:
            index = np.random.randint(history_len, count)
            if index not in indexes:
                if not index >= pos > index - history_len:
                    if not terminals[index - history_len:index].any():
                        indexes.append(index)
        return indexes

    def minibatch(self, size):
        if False:
            i = 10
            return i + 15
        ' Generate a minibatch with the number of samples specified by the size parameter.\n\n        Attributes:\n            size (int): Minibatch size\n\n        Returns:\n            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]\n        '
        indexes = self.sample(size)
        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]
        return (pre_states, actions, post_states, rewards, dones)

    def get_state(self, index):
        if False:
            i = 10
            return i + 15
        "\n        Return the specified state with the replay memory. A state consists of\n        the last `history_length` perceptions.\n\n        Attributes:\n            index (int): State's index\n\n        Returns:\n            State at specified index (Tensor[history_length, input_shape...])\n        "
        if self._count == 0:
            raise IndexError('Empty Memory')
        index %= self._count
        history_length = self._history_length
        if index >= history_length:
            return self._states[index - (history_length - 1):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        if False:
            return 10
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        if False:
            while True:
                i = 10
        ' Underlying buffer with N previous states stacked along first axis\n\n        Returns:\n            Tensor[shape]\n        '
        return self._buffer

    def append(self, state):
        if False:
            while True:
                i = 10
        ' Append state to the history\n\n        Attributes:\n            state (Tensor) : The state to append to the memory\n        '
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        if False:
            i = 10
            return i + 15
        ' Reset the memory. Underlying buffer set all indexes to 0\n\n        '
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        if False:
            while True:
                i = 10
        self._start = start
        self._stop = end
        self._steps = steps
        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        if False:
            return 10
        '\n        Select a random action out of `num_actions` possibilities.\n\n        Attributes:\n            num_actions (int): Number of actions available\n        '
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        if False:
            while True:
                i = 10
        ' Compute the epsilon parameter according to the specified step\n\n        Attributes:\n            step (int)\n        '
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        if False:
            for i in range(10):
                print('nop')
        ' Commodity method indicating if the agent should explore\n\n        Attributes:\n            step (int) : Current step\n\n        Returns:\n             bool : True if exploring, False otherwise\n        '
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    if False:
        for i in range(10):
            print('nop')
    ' Compute the Huber Loss as part of the model graph\n\n    Huber Loss is more robust to outliers. It is defined as:\n     if |y - y_hat| < delta :\n        0.5 * (y - y_hat)**2\n    else :\n        delta * |y - y_hat| - 0.5 * delta**2\n\n    Attributes:\n        y (Tensor[-1, 1]): Target value\n        y_hat(Tensor[-1, 1]): Estimated value\n        delta (float): Outliers threshold\n\n    Returns:\n        CNTK Graph Node\n    '
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)
    less_than = 0.5 * square(error)
    more_than = delta * abs_error - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)
    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """

    def __init__(self, input_shape, nb_actions, gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000), learning_rate=0.00025, momentum=0.95, minibatch_size=32, memory_size=500000, train_after=200000, train_interval=4, target_update_interval=10000, monitor=True):
        if False:
            print('Hello World!')
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma
        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval
        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0
        (self._episode_rewards, self._episode_q_means, self._episode_q_stddev) = ([], [], [])
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([Convolution2D((8, 8), 16, strides=4), Convolution2D((4, 4), 32, strides=2), Convolution2D((3, 3), 32, strides=1), Dense(256, init=he_uniform(scale=0.01)), Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))])
        self._action_value_net.update_signature(Tensor[input_shape])
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            if False:
                for i in range(10):
                    print('nop')
            return element_select(terminals, rewards, gamma * reduce_max(self._target_net(post_states), axis=0) + rewards)

        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions], post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            if False:
                return 10
            q_targets = compute_q_targets(post_states, rewards, terminals)
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)
            return huber_loss(q_targets, q_acted, 1.0)
        lr_schedule = learning_parameter_schedule(learning_rate)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule, momentum=m_schedule, variance_momentum=vm_schedule)
        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

    def act(self, state):
        if False:
            for i in range(10):
                print('nop')
        ' This allows the agent to select the next action to perform in regard of the current state of the environment.\n        It follows the terminology used in the Nature paper.\n\n        Attributes:\n            state (Tensor[input_shape]): The current environment state\n\n        Returns: Int >= 0 : Next action to do\n        '
        self._history.append(state)
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(env_with_history.reshape((1,) + env_with_history.shape))
            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))
            action = q_values.argmax()
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        if False:
            return 10
        ' This allows the agent to observe the output of doing the action it selected through act() on the old_state\n\n        Attributes:\n            old_state (Tensor[input_shape]): Previous environment state\n            action (int): Action done by the agent\n            reward (float): Reward for doing this action in the old_state environment\n            done (bool): Indicate if the action has terminated the environment\n        '
        self._episode_rewards.append(reward)
        if done:
            if self._metrics_writer is not None:
                self._plot_metrics()
            (self._episode_rewards, self._episode_q_means, self._episode_q_stddev) = ([], [], [])
            self._history.reset()
        self._memory.append(old_state, action, reward, done)

    def train(self):
        if False:
            print('Hello World!')
        ' This allows the agent to train itself to better understand the environment dynamics.\n        The agent will compute the expected reward for the state(t+1)\n        and update the expected reward at step t according to this.\n\n        The target expectation is computed through the Target Network, which is a more stable version\n        of the Action Value Network for increasing training stability.\n\n        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.\n        '
        agent_step = self._num_actions_taken
        if agent_step >= self._train_after:
            if agent_step % self._train_interval == 0:
                (pre_states, actions, post_states, rewards, terminals) = self._memory.minibatch(self._minibatch_size)
                self._trainer.train_minibatch(self._trainer.loss_function.argument_map(pre_states=pre_states, actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions), post_states=post_states, rewards=rewards, terminals=terminals))
                if agent_step % self._target_update_interval == 0:
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)

    def _plot_metrics(self):
        if False:
            print('Hello World!')
        'Plot current buffers accumulated values to visualize agent learning\n        '
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)
        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)
        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)

def as_ale_input(environment):
    if False:
        while True:
            i = 10
    'Convert the Atari environment RGB output (210, 160, 3) to an ALE one (84, 84).\n    We first convert the image to a gray scale image, and resize it.\n\n    Attributes:\n        environment (Tensor[input_shape]): Environment to be converted\n\n    Returns:\n         Tensor[84, 84] : Environment converted\n    '
    from PIL import Image
    return np.array(Image.fromarray(environment).convert('L').resize((84, 84)))
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of epochs to run (epoch = 250k actions')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Flag for enabling Tensorboard')
    parser.add_argument('env', default='Pong-v3', type=str, metavar='N', nargs='?', help='Gym Atari environment to run')
    args = parser.parse_args()
    env = gym.make(args.env)
    agent = DeepQAgent((4, 84, 84), env.action_space.n, monitor=args.plot)
    current_step = 0
    max_steps = args.epoch * 250000
    current_state = as_ale_input(env.reset())
    while current_step < max_steps:
        action = agent.act(current_state)
        (new_state, reward, done, _) = env.step(action)
        new_state = as_ale_input(new_state)
        reward = np.clip(reward, -1, 1)
        agent.observe(current_state, action, reward, done)
        agent.train()
        current_state = new_state
        if done:
            current_state = as_ale_input(env.reset())
        current_step += 1