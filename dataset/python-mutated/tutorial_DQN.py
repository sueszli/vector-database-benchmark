"""
Deep Q-Network Q(a, s)
-----------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).
Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)
See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
Reference
----------
original paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327
Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py
Environment
-----------
# The FrozenLake v0 environment
https://gym.openai.com/envs/FrozenLake-v0
The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially
depends on the chosen direction. The agent is rewarded for finding a walkable
path to a goal tile.
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward
of 1 if you reach the goal, and zero otherwise.
Prerequisites
--------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
To run
-------
python tutorial_DQN.py --train/test
"""
import argparse
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()
tl.logging.set_verbosity(tl.logging.DEBUG)
env_id = 'FrozenLake-v0'
alg_name = 'DQN'
lambd = 0.99
e = 0.1
num_episodes = 10000
render = False

def to_one_hot(i, n_classes=None):
    if False:
        for i in range(10):
            print('nop')
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a

def get_model(inputs_shape):
    if False:
        while True:
            i = 10
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name='Q-Network')

def save_ckpt(model):
    if False:
        i = 10
        return i + 15
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)

def load_ckpt(model):
    if False:
        i = 10
        return i + 15
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)
if __name__ == '__main__':
    qnetwork = get_model([None, 16])
    qnetwork.train()
    train_weights = qnetwork.trainable_weights
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    env = gym.make(env_id)
    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            if render:
                env.render()
            for j in range(99):
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                (s1, r, d, _) = env.step(a[0])
                if render:
                    env.render()
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + lambd * maxQ1
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))
                rAll += r
                s = s1
                if d == True:
                    e = 1.0 / (i / 50 + 10)
                    break
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'.format(i, num_episodes, rAll, time.time() - t0))
            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)
        save_ckpt(qnetwork)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))
    if args.test:
        load_ckpt(qnetwork)
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            if render:
                env.render()
            for j in range(99):
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)
                (s1, r, d, _) = env.step(a[0])
                rAll += r
                s = s1
                if render:
                    env.render()
                if d:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'.format(i, num_episodes, rAll, time.time() - t0))