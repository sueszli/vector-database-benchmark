"""
Asynchronous Advantage Actor Critic (A3C) with Continuous Action Space.

Actor Critic History
----------------------
A3C > DDPG (for continuous action space) > AC

Advantage
----------
Train faster and more stable than AC.

Disadvantage
-------------
Have bias.

Reference
----------
Original Paper: https://arxiv.org/pdf/1602.01783.pdf
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_BipedalWalker/A3C.py

Environment
-----------
BipedalWalker-v2 : https://gym.openai.com/envs/BipedalWalker-v2

Reward is given for moving forward, total 300+ points up to the far end.
If the robot falls, it gets -100. Applying motor torque costs a small amount of
points, more optimal agent will get better score. State consists of hull angle
speed, angular velocity, horizontal speed, vertical speed, position of joints
and joints angular speed, legs contact with ground, and 10 lidar rangefinder
measurements. There's no coordinates in the state vector.

Prerequisites
--------------
tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0
&&
pip install box2d box2d-kengz --user

To run
------
python tutorial_A3C.py --train/test

"""
import argparse
import multiprocessing
import os
import threading
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
tfd = tfp.distributions
tl.logging.set_verbosity(tl.logging.DEBUG)
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()
ENV_ID = 'BipedalWalker-v2'
RANDOM_SEED = 2
RENDER = False
ALG_NAME = 'A3C'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 15000
TEST_EPISODES = 10
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.005
LR_A = 5e-05
LR_C = 0.0001
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

class ACNet(object):

    def __init__(self, scope):
        if False:
            print('Hello World!')
        self.scope = scope
        w_init = tf.keras.initializers.glorot_normal(seed=None)

        def get_actor(input_shape):
            if False:
                i = 10
                return i + 15
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')(nn)
                mu = tl.layers.Dense(n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')(nn)
                sigma = tl.layers.Dense(n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')(nn)
            return tl.models.Model(inputs=ni, outputs=[mu, sigma], name=scope + '/Actor')
        self.actor = get_actor([None, N_S])
        self.actor.train()

        def get_critic(input_shape):
            if False:
                while True:
                    i = 10
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='lc2')(nn)
                v = tl.layers.Dense(n_units=1, W_init=w_init, name='v')(nn)
            return tl.models.Model(inputs=ni, outputs=v, name=scope + '/Critic')
        self.critic = get_critic([None, N_S])
        self.critic.train()

    @tf.function
    def update_global(self, buffer_s, buffer_a, buffer_v_target, globalAC):
        if False:
            i = 10
            return i + 15
        ' update the global critic '
        with tf.GradientTape() as tape:
            self.v = self.critic(buffer_s)
            self.v_target = buffer_v_target
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            self.c_loss = tf.reduce_mean(tf.square(td))
        self.c_grads = tape.gradient(self.c_loss, self.critic.trainable_weights)
        OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic.trainable_weights))
        ' update the global actor '
        with tf.GradientTape() as tape:
            (self.mu, self.sigma) = self.actor(buffer_s)
            self.test = self.sigma[0]
            (self.mu, self.sigma) = (self.mu * A_BOUND[1], self.sigma + 1e-05)
            normal_dist = tfd.Normal(self.mu, self.sigma)
            self.a_his = buffer_a
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * td
            entropy = normal_dist.entropy()
            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
        self.a_grads = tape.gradient(self.a_loss, self.actor.trainable_weights)
        OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor.trainable_weights))
        return self.test

    @tf.function
    def pull_global(self, globalAC):
        if False:
            for i in range(10):
                print('nop')
        for (l_p, g_p) in zip(self.actor.trainable_weights, globalAC.actor.trainable_weights):
            l_p.assign(g_p)
        for (l_p, g_p) in zip(self.critic.trainable_weights, globalAC.critic.trainable_weights):
            l_p.assign(g_p)

    def get_action(self, s, greedy=False):
        if False:
            for i in range(10):
                print('nop')
        s = s[np.newaxis, :]
        (self.mu, self.sigma) = self.actor(s)
        with tf.name_scope('wrap_a_out'):
            (self.mu, self.sigma) = (self.mu * A_BOUND[1], self.sigma + 1e-05)
        if greedy:
            return self.mu.numpy()[0]
        normal_dist = tfd.Normal(self.mu, self.sigma)
        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)
        return self.A.numpy()[0]

    def save(self):
        if False:
            while True:
                i = 10
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.actor.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz(self.critic.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.actor)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.critic)

class Worker(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.env = gym.make(ENV_ID)
        self.name = name
        self.AC = ACNet(name)

    def work(self, globalAC):
        if False:
            for i in range(10):
                print('nop')
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        (buffer_s, buffer_a, buffer_r) = ([], [], [])
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                if RENDER and self.name == 'Worker_0' and (total_step % 30 == 0):
                    self.env.render()
                s = s.astype('float32')
                a = self.AC.get_action(s)
                (s_, r, done, _info) = self.env.step(a)
                s_ = s_.astype('float32')
                if r == -100:
                    r = -2
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s = tf.convert_to_tensor(np.vstack(buffer_s))
                    buffer_a = tf.convert_to_tensor(np.vstack(buffer_a))
                    buffer_v_target = tf.convert_to_tensor(np.vstack(buffer_v_target).astype('float32'))
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target, globalAC)
                    (buffer_s, buffer_a, buffer_r) = ([], [], [])
                    self.AC.pull_global(globalAC)
                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print('Training  | {}, Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(self.name, GLOBAL_EP, MAX_GLOBAL_EP, ep_r, time.time() - T0))
                    GLOBAL_EP += 1
                    break
if __name__ == '__main__':
    env = gym.make(ENV_ID)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]
    A_BOUND = [env.action_space.low, env.action_space.high]
    A_BOUND[0] = A_BOUND[0].reshape(1, N_A)
    A_BOUND[1] = A_BOUND[1].reshape(1, N_A)
    with tf.device('/cpu:0'):
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
    T0 = time.time()
    if args.train:
        with tf.device('/cpu:0'):
            OPT_A = tf.optimizers.RMSprop(LR_A, name='RMSPropA')
            OPT_C = tf.optimizers.RMSprop(LR_C, name='RMSPropC')
            workers = []
            for i in range(N_WORKERS):
                i_name = 'Worker_%i' % i
                workers.append(Worker(i_name))
        COORD = tf.train.Coordinator()
        worker_threads = []
        for worker in workers:
            job = lambda : worker.work(GLOBAL_AC)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        GLOBAL_AC.save()
        plt.plot(GLOBAL_RUNNING_R)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
    if args.test:
        GLOBAL_AC.load()
        for episode in range(TEST_EPISODES):
            s = env.reset()
            episode_reward = 0
            while True:
                env.render()
                s = s.astype('float32')
                a = GLOBAL_AC.get_action(s, greedy=True)
                (s, r, d, _) = env.step(a)
                episode_reward += r
                if d:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(episode + 1, TEST_EPISODES, episode_reward, time.time() - T0))