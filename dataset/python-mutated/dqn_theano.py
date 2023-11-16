from __future__ import print_function, division
from builtins import range
import copy
import gym
import os
import sys
import random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4

def rgb2gray(rgb):
    if False:
        print('Hello World!')
    (r, g, b) = (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])
    gray = 0.2989 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def downsample_image(A):
    if False:
        print('Hello World!')
    B = A[34:194]
    B = rgb2gray(B)
    B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
    return B

def update_state(state, obs):
    if False:
        print('Hello World!')
    obs_small = downsample_image(obs)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)

class ReplayMemory:

    def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, agent_history_length=4, batch_size=32):
        if False:
            for i in range(10):
                print('nop')
        '\n    Args:\n        size: Integer, Number of stored transitions\n        frame_height: Integer, Height of a frame of an Atari game\n        frame_width: Integer, Width of a frame of an Atari game\n        agent_history_length: Integer, Number of frames stacked together to create a state\n        batch_size: Integer, Number of transitions returned in a minibatch\n    '
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        if False:
            for i in range(10):
                print('nop')
        '\n    Args:\n        action: An integer-encoded action\n        frame: One grayscale frame of the game\n        reward: reward the agend received for performing an action\n        terminal: A bool stating whether the episode terminated\n    '
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if False:
            print('Hello World!')
        if self.count is 0:
            raise ValueError('The replay memory is empty!')
        if index < self.agent_history_length - 1:
            raise ValueError('Index must be min 3')
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        if False:
            i = 10
            return i + 15
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if False:
            i = 10
            return i + 15
        '\n    Returns a minibatch of self.batch_size transitions\n    '
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        self._get_valid_indices()
        for (i, idx) in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return (self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices])

def init_filter(shape):
    if False:
        while True:
            i = 10
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[1:]))
    return w.astype(np.float32)

def adam(cost, params, lr0=1e-05, beta1=0.9, beta2=0.999, eps=1e-08):
    if False:
        while True:
            i = 10
    lr0 = np.float32(lr0)
    beta1 = np.float32(beta1)
    beta2 = np.float32(beta2)
    eps = np.float32(eps)
    one = np.float32(1)
    zero = np.float32(0)
    grads = T.grad(cost, params)
    updates = []
    time = theano.shared(zero)
    new_time = time + one
    updates.append((time, new_time))
    lr = lr0 * T.sqrt(one - beta2 ** new_time) / (one - beta1 ** new_time)
    for (p, g) in zip(params, grads):
        m = theano.shared(p.get_value() * zero)
        v = theano.shared(p.get_value() * zero)
        new_m = beta1 * m + (one - beta1) * g
        new_v = beta2 * v + (one - beta2) * g * g
        new_p = p - lr * new_m / (T.sqrt(new_v) + eps)
        updates.append((m, new_m))
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates

class ConvLayer(object):

    def __init__(self, mi, mo, filtsz=5, stride=2, f=T.nnet.relu):
        if False:
            print('Hello World!')
        sz = (mo, mi, filtsz, filtsz)
        W0 = init_filter(sz)
        self.W = theano.shared(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = theano.shared(b0)
        self.stride = (stride, stride)
        self.params = [self.W, self.b]
        self.f = f

    def forward(self, X):
        if False:
            for i in range(10):
                print('nop')
        conv_out = conv2d(input=X, filters=self.W, subsample=self.stride, border_mode='valid')
        return self.f(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class HiddenLayer:

    def __init__(self, M1, M2, f=T.nnet.relu):
        if False:
            while True:
                i = 10
        W = np.random.randn(M1, M2) * np.sqrt(2 / M1)
        self.W = theano.shared(W.astype(np.float32))
        self.b = theano.shared(np.zeros(M2).astype(np.float32))
        self.params = [self.W, self.b]
        self.f = f

    def forward(self, X):
        if False:
            return 10
        a = X.dot(self.W) + self.b
        return self.f(a)

class DQN:

    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes):
        if False:
            print('Hello World!')
        self.K = K
        X = T.ftensor4('X')
        G = T.fvector('G')
        actions = T.ivector('actions')
        self.conv_layers = []
        num_input_filters = 4
        current_size = IM_SIZE
        for (num_output_filters, filtersz, stride) in conv_layer_sizes:
            layer = ConvLayer(num_input_filters, num_output_filters, filtersz, stride)
            current_size = (current_size + stride - 1) // stride
            self.conv_layers.append(layer)
            num_input_filters = num_output_filters
        Z = X / 255.0
        for layer in self.conv_layers:
            Z = layer.forward(Z)
        conv_out = Z.flatten(ndim=2)
        conv_out_op = theano.function(inputs=[X], outputs=conv_out, allow_input_downcast=True)
        test = conv_out_op(np.random.randn(1, 4, IM_SIZE, IM_SIZE))
        flattened_ouput_size = test.shape[1]
        self.layers = []
        M1 = flattened_ouput_size
        print('flattened_ouput_size:', flattened_ouput_size)
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)
        self.params = []
        for layer in self.conv_layers + self.layers:
            self.params += layer.params
        Z = conv_out
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        selected_action_values = Y_hat[T.arange(actions.shape[0]), actions]
        cost = T.mean((G - selected_action_values) ** 2)
        updates = adam(cost, self.params)
        self.train_op = theano.function(inputs=[X, G, actions], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict_op = theano.function(inputs=[X], outputs=Y_hat, allow_input_downcast=True)

    def copy_from(self, other):
        if False:
            while True:
                i = 10
        my_params = self.params
        other_params = other.params
        for (p, q) in zip(my_params, other_params):
            actual = q.get_value()
            p.set_value(actual)

    def predict(self, X):
        if False:
            while True:
                i = 10
        return self.predict_op(X)

    def update(self, states, actions, targets):
        if False:
            return 10
        return self.train_op(states, targets, actions)

    def sample_action(self, x, eps):
        if False:
            while True:
                i = 10
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    if False:
        i = 10
        return i + 15
    (states, actions, rewards, next_states, dones) = experience_replay_buffer.get_minibatch()
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
    loss = model.update(states, actions, targets)
    return loss

def play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size, epsilon, epsilon_change, epsilon_min):
    if False:
        while True:
            i = 10
    t0 = datetime.now()
    obs = env.reset()
    obs_small = downsample_image(obs)
    state = np.stack([obs_small] * 4, axis=0)
    loss = None
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0
    done = False
    while not done:
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print('Copied model parameters to target network. total_t = %s, period = %s' % (total_t, TARGET_UPDATE_PERIOD))
        action = model.sample_action(state, epsilon)
        (obs, reward, done, _) = env.step(action)
        obs_small = downsample_image(obs)
        next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
        episode_reward += reward
        experience_replay_buffer.add_experience(action, obs_small, reward, done)
        t0_2 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
        state = next_state
        total_t += 1
        epsilon = max(epsilon - epsilon_change, epsilon_min)
    return (total_t, episode_reward, datetime.now() - t0, num_steps_in_episode, total_time_training / num_steps_in_episode, epsilon)

def smooth(x):
    if False:
        return 10
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:i + 1].sum()) / (i - start + 1)
    return y
if __name__ == '__main__':
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_sz = 32
    num_episodes = 5000
    total_t = 0
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)
    step_counts = np.zeros(num_episodes)
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000
    env = gym.envs.make('Breakout-v0')
    model = DQN(K=K, conv_layer_sizes=conv_layer_sizes, hidden_layer_sizes=hidden_layer_sizes)
    target_model = DQN(K=K, conv_layer_sizes=conv_layer_sizes, hidden_layer_sizes=hidden_layer_sizes)
    print('Populating experience replay buffer...')
    obs = env.reset()
    obs_small = downsample_image(obs)
    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(K)
        (obs, reward, done, _) = env.step(action)
        obs_small = downsample_image(obs)
        experience_replay_buffer.add_experience(action, obs_small, reward, done)
        if done:
            obs = env.reset()
    t0 = datetime.now()
    for i in range(num_episodes):
        (total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon) = play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_sz, epsilon, epsilon_change, epsilon_min)
        episode_rewards[i] = episode_reward
        step_counts[i] = num_steps_in_episode
        last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
        last_100_avg_steps = step_counts[max(0, i - 100):i + 1].mean()
        print('Episode:', i, 'Duration:', duration, 'Num steps:', num_steps_in_episode, 'Reward:', episode_reward, 'Training time per step:', '%.3f' % time_per_step, 'Avg Reward (Last 100):', '%.3f' % last_100_avg, 'Avg Steps (Last 100):', '%.1f' % last_100_avg_steps, 'Epsilon:', '%.3f' % epsilon)
        sys.stdout.flush()
    print('Total duration:', datetime.now() - t0)
    y = smooth(episode_rewards)
    plt.plot(episode_rewards, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()