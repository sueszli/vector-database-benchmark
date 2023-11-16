import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.figure()
import gym, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from collections import namedtuple
from torch.utils.data import SubsetRandomSampler, BatchSampler
env = gym.make('CartPole-v1')
env.seed(2222)
tf.random.set_seed(2222)
np.random.seed(2222)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
gamma = 0.98
epsilon = 0.2
batch_size = 32
env = gym.make('CartPole-v0').unwrapped
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(keras.Model):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')

    def call(self, inputs):
        if False:
            while True:
                i = 10
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1)
        return x

class Critic(keras.Model):

    def __init__(self):
        if False:
            print('Hello World!')
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs):
        if False:
            return 10
        x = tf.nn.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x

class PPO:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(PPO, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = []
        self.actor_optimizer = optimizers.Adam(0.001)
        self.critic_optimizer = optimizers.Adam(0.003)

    def select_action(self, s):
        if False:
            print('Hello World!')
        s = tf.constant(s, dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        prob = self.actor(s)
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a)
        return (a, float(prob[0][a]))

    def get_value(self, s):
        if False:
            print('Hello World!')
        s = tf.constant(s, dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        v = self.critic(s)[0]
        return float(v)

    def store_transition(self, transition):
        if False:
            print('Hello World!')
        self.buffer.append(transition)

    def optimize(self):
        if False:
            while True:
                i = 10
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1, 1])
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + gamma * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)
        for _ in range(round(10 * len(self.buffer) / batch_size)):
            index = np.random.choice(np.arange(len(self.buffer)), batch_size, replace=False)
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v
                advantage = tf.stop_gradient(delta)
                a = tf.gather(action, index, axis=0)
                pi = self.actor(tf.gather(state, index, axis=0))
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                indices = tf.concat([indices, a], axis=1)
                pi_a = tf.gather_nd(pi, indices)
                pi_a = tf.expand_dims(pi_a, axis=1)
                ratio = pi_a / tf.gather(old_action_log_prob, index, axis=0)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                value_loss = losses.MSE(v_target, v)
            grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            grads = tape2.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        self.buffer = []

def main():
    if False:
        print('Hello World!')
    agent = PPO()
    returns = []
    total = 0
    for i_epoch in range(500):
        state = env.reset()
        for t in range(500):
            (action, action_prob) = agent.select_action(state)
            (next_state, reward, done, _) = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            state = next_state
            total += reward
            if done:
                if len(agent.buffer) >= batch_size:
                    agent.optimize()
                break
        if i_epoch % 20 == 0:
            returns.append(total / 20)
            total = 0
            print(i_epoch, returns[-1])
    print(np.array(returns))
    plt.figure()
    plt.plot(np.arange(len(returns)) * 20, np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('ppo-tf-cartpole.svg')
if __name__ == '__main__':
    main()
    print('end')