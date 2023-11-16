import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.figure()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
tf.random.set_seed(1231)
np.random.seed(1231)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

class ActorCritic(keras.Model):

    def __init__(self, state_size, action_size):
        if False:
            for i in range(10):
                print('nop')
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(128, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(128, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v = self.dense2(inputs)
        values = self.values(v)
        return (logits, values)

def record(episode, episode_reward, worker_idx, global_ep_reward, result_queue, total_loss, num_steps):
    if False:
        while True:
            i = 10
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(f'{episode} | Average Reward: {int(global_ep_reward)} | Episode Reward: {int(episode_reward)} | Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | Steps: {num_steps} | Worker: {worker_idx}')
    result_queue.put(global_ep_reward)
    return global_ep_reward

class Memory:

    def __init__(self):
        if False:
            print('Hello World!')
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        if False:
            for i in range(10):
                print('nop')
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.states = []
        self.actions = []
        self.rewards = []

class Agent:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.opt = optimizers.Adam(0.001)
        self.server = ActorCritic(4, 2)
        self.server(tf.random.normal((2, 4)))

    def train(self):
        if False:
            i = 10
            return i + 15
        res_queue = Queue()
        workers = [Worker(self.server, self.opt, res_queue, i) for i in range(multiprocessing.cpu_count())]
        for (i, worker) in enumerate(workers):
            print('Starting worker {}'.format(i))
            worker.start()
        returns = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                returns.append(reward)
            else:
                break
        [w.join() for w in workers]
        print(returns)
        plt.figure()
        plt.plot(np.arange(len(returns)), returns)
        plt.xlabel('回合数')
        plt.ylabel('总回报')
        plt.savefig('a3c-tf-cartpole.svg')

class Worker(threading.Thread):

    def __init__(self, server, opt, result_queue, idx):
        if False:
            while True:
                i = 10
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.server = server
        self.opt = opt
        self.client = ActorCritic(4, 2)
        self.worker_idx = idx
        self.env = gym.make('CartPole-v1').unwrapped
        self.ep_loss = 0.0

    def run(self):
        if False:
            i = 10
            return i + 15
        mem = Memory()
        for epi_counter in range(500):
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            while not done:
                (logits, _) = self.client(tf.constant(current_state[None, :], dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                action = np.random.choice(2, p=probs.numpy()[0])
                (new_state, reward, done, _) = self.env.step(action)
                ep_reward += reward
                mem.store(current_state, action, reward)
                ep_steps += 1
                current_state = new_state
                if ep_steps >= 500 or done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem)
                    grads = tape.gradient(total_loss, self.client.trainable_weights)
                    self.opt.apply_gradients(zip(grads, self.server.trainable_weights))
                    self.client.set_weights(self.server.get_weights())
                    mem.clear()
                    self.result_queue.put(ep_reward)
                    print(self.worker_idx, ep_reward)
                    break
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if False:
            while True:
                i = 10
        if done:
            reward_sum = 0.0
        else:
            reward_sum = self.client(tf.constant(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        (logits, values) = self.client(tf.constant(np.vstack(memory.states), dtype=tf.float32))
        advantage = tf.constant(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        value_loss = advantage ** 2
        policy = tf.nn.softmax(logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss = policy_loss * tf.stop_gradient(advantage)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        policy_loss = policy_loss - 0.01 * entropy
        total_loss = tf.reduce_mean(0.5 * value_loss + policy_loss)
        return total_loss
if __name__ == '__main__':
    master = Agent()
    master.train()