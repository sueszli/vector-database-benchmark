from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning import plot_running_avg, FeatureTransformer

class HiddenLayer:

    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
        if False:
            return 10
        if zeros:
            W = np.zeros((M1, M2)).astype(np.float32)
            self.W = tf.Variable(W)
        else:
            self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if False:
            while True:
                i = 10
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class PolicyModel:

    def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):
        if False:
            return 10
        self.ft = ft
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_var = hidden_layer_sizes_var
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)
        self.var_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1, M2)
            self.var_layers.append(layer)
            M1 = M2
        layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.var_layers.append(layer)
        self.params = []
        for layer in self.mean_layers + self.var_layers:
            self.params += layer.params
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        def get_output(layers):
            if False:
                print('Hello World!')
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
            return tf.reshape(Z, [-1])
        mean = get_output(self.mean_layers)
        std = get_output(self.var_layers) + 0.0001
        norm = tf.contrib.distributions.Normal(mean, std)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

    def set_session(self, session):
        if False:
            print('Hello World!')
        self.session = session

    def init_vars(self):
        if False:
            while True:
                i = 10
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def predict(self, X):
        if False:
            while True:
                i = 10
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        if False:
            print('Hello World!')
        p = self.predict(X)[0]
        return p

    def copy(self):
        if False:
            while True:
                i = 10
        clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_mean)
        clone.set_session(self.session)
        clone.init_vars()
        clone.copy_from(self)
        return clone

    def copy_from(self, other):
        if False:
            i = 10
            return i + 15
        ops = []
        my_params = self.params
        other_params = other.params
        for (p, q) in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def perturb_params(self):
        if False:
            print('Hello World!')
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                op = p.assign(noise)
            else:
                op = p.assign(v + noise)
            ops.append(op)
        self.session.run(ops)

def play_one(env, pmodel, gamma):
    if False:
        while True:
            i = 10
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 2000:
        action = pmodel.sample_action(observation)
        (observation, reward, done, info) = env.step([action])
        totalreward += reward
        iters += 1
    return totalreward

def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False):
    if False:
        while True:
            i = 10
    totalrewards = np.empty(T)
    for i in range(T):
        totalrewards[i] = play_one(env, pmodel, gamma)
        if print_iters:
            print(i, 'avg so far:', totalrewards[:i + 1].mean())
    avg_totalrewards = totalrewards.mean()
    print('avg totalrewards:', avg_totalrewards)
    return avg_totalrewards

def random_search(env, pmodel, gamma):
    if False:
        print('Hello World!')
    totalrewards = []
    best_avg_totalreward = float('-inf')
    best_pmodel = pmodel
    num_episodes_per_param_test = 3
    for t in range(100):
        tmp_pmodel = best_pmodel.copy()
        tmp_pmodel.perturb_params()
        avg_totalrewards = play_multiple_episodes(env, num_episodes_per_param_test, tmp_pmodel, gamma)
        totalrewards.append(avg_totalrewards)
        if avg_totalrewards > best_avg_totalreward:
            best_pmodel = tmp_pmodel
            best_avg_totalreward = avg_totalrewards
    return (totalrewards, best_pmodel)

def main():
    if False:
        while True:
            i = 10
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env, n_components=100)
    D = ft.dimensions
    pmodel = PolicyModel(ft, D, [], [])
    session = tf.InteractiveSession()
    pmodel.set_session(session)
    pmodel.init_vars()
    gamma = 0.99
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    (totalrewards, pmodel) = random_search(env, pmodel, gamma)
    print('max reward:', np.max(totalrewards))
    avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
    print('avg reward over 100 episodes with best models:', avg_totalrewards)
    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()
if __name__ == '__main__':
    main()