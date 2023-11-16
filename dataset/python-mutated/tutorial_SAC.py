""" 
Soft Actor-Critic (SAC)
------------------
Actor policy in SAC is stochastic, with off-policy training. 
And 'soft' in SAC indicates the trade-off between the entropy and expected return. 
The additional consideration of entropy term helps with more explorative policy.
And this implementation contains an automatic update for the entropy factor.
This version of Soft Actor-Critic (SAC) implementation contains 5 networks: 
2 Q net, 2 target Q net, 1 policy net.
It uses alpha loss.
Reference
---------
paper: https://arxiv.org/pdf/1812.05905.pdf
Environment
---
Openai Gym Pendulum-v0, continuous action space
https://gym.openai.com/envs/Pendulum-v0/
Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
&&
pip install box2d box2d-kengz --user
To run
------
python tutorial_SAC.py --train/test
"""
import argparse
import os
import random
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()
ENV_ID = 'Pendulum-v0'
RANDOM_SEED = 2
RENDER = False
ALG_NAME = 'SAC'
TRAIN_EPISODES = 100
TEST_EPISODES = 10
MAX_STEPS = 200
EXPLORE_STEPS = 100
BATCH_SIZE = 256
HIDDEN_DIM = 32
UPDATE_ITR = 3
SOFT_Q_LR = 0.0003
POLICY_LR = 0.0003
ALPHA_LR = 0.0003
POLICY_TARGET_UPDATE_INTERVAL = 3
REWARD_SCALE = 1.0
REPLAY_BUFFER_SIZE = 500000.0
AUTO_ENTROPY = True

class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        if False:
            return 10
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if False:
            i = 10
            return i + 15
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, BATCH_SIZE):
        if False:
            while True:
                i = 10
        batch = random.sample(self.buffer, BATCH_SIZE)
        (state, action, reward, next_state, done) = map(np.stack, zip(*batch))
        ' \n        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;\n        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;\n        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;\n        np.stack((1,2)) => array([1, 2])\n        '
        return (state, action, reward, next_state, done)

    def __len__(self):
        if False:
            return 10
        return len(self.buffer)

class SoftQNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=0.003):
        if False:
            i = 10
            return i + 15
        super(SoftQNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        w_init = tf.keras.initializers.glorot_normal(seed=None)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class PolicyNetwork(Model):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1.0, init_w=0.003, log_std_min=-20, log_std_max=2):
        if False:
            i = 10
            return i + 15
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        w_init = tf.keras.initializers.glorot_normal(seed=None)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.mean_linear = Dense(n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_mean')
        self.log_std_linear = Dense(n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_logstd')
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        if False:
            print('Hello World!')
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return (mean, log_std)

    def evaluate(self, state, epsilon=1e-06):
        if False:
            print('Hello World!')
        ' generate action with state for calculating gradients '
        state = state.astype(np.float32)
        (mean, log_std) = self.forward(state)
        std = tf.math.exp(log_std)
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = tf.math.tanh(mean + std * z)
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - tf.math.log(1.0 - action_0 ** 2 + epsilon) - np.log(self.action_range)
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, np.newaxis]
        return (action, log_prob, z, mean, log_std)

    def get_action(self, state, greedy=False):
        if False:
            i = 10
            return i + 15
        ' generate action with state for interaction with envronment '
        (mean, log_std) = self.forward([state])
        std = tf.math.exp(log_std)
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action = self.action_range * tf.math.tanh(mean + std * z)
        action = self.action_range * tf.math.tanh(mean) if greedy else action
        return action.numpy()[0]

    def sample_action(self):
        if False:
            while True:
                i = 10
        ' generate random actions for exploration '
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()

class SAC:

    def __init__(self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, SOFT_Q_LR=0.0003, POLICY_LR=0.0003, ALPHA_LR=0.0003):
        if False:
            return 10
        self.replay_buffer = replay_buffer
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()
        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        self.alpha = tf.math.exp(self.log_alpha)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)
        self.soft_q_optimizer1 = tf.optimizers.Adam(SOFT_Q_LR)
        self.soft_q_optimizer2 = tf.optimizers.Adam(SOFT_Q_LR)
        self.policy_optimizer = tf.optimizers.Adam(POLICY_LR)
        self.alpha_optimizer = tf.optimizers.Adam(ALPHA_LR)

    def target_ini(self, net, target_net):
        if False:
            return 10
        ' hard-copy update for initializing target networks '
        for (target_param, param) in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        if False:
            for i in range(10):
                print('nop')
        ' soft update the target net with Polyak averaging '
        for (target_param, param) in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)
        return target_net

    def update(self, batch_size, reward_scale=10.0, auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=0.01):
        if False:
            for i in range(10):
                print('nop')
        ' update all networks in SAC '
        (state, action, reward, next_state, done) = self.replay_buffer.sample(batch_size)
        reward = reward[:, np.newaxis]
        done = done[:, np.newaxis]
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (np.std(reward, axis=0) + 1e-06)
        (new_next_action, next_log_prob, _, _, _) = self.policy_net.evaluate(next_state)
        target_q_input = tf.concat([next_state, new_next_action], 1)
        target_q_min = tf.minimum(self.target_soft_q_net1(target_q_input), self.target_soft_q_net2(target_q_input)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min
        q_input = tf.concat([state, action], 1)
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.soft_q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value1, target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.soft_q_net1.trainable_weights)
        self.soft_q_optimizer1.apply_gradients(zip(q1_grad, self.soft_q_net1.trainable_weights))
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.soft_q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value2, target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.soft_q_net2.trainable_weights)
        self.soft_q_optimizer2.apply_gradients(zip(q2_grad, self.soft_q_net2.trainable_weights))
        with tf.GradientTape() as p_tape:
            (new_action, log_prob, z, mean, log_std) = self.policy_net.evaluate(state)
            new_q_input = tf.concat([state, new_action], 1)
            ' implementation 1 '
            predicted_new_q_value = tf.minimum(self.soft_q_net1(new_q_input), self.soft_q_net2(new_q_input))
            policy_loss = tf.reduce_mean(self.alpha * log_prob - predicted_new_q_value)
        p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))
        if auto_entropy is True:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean(self.log_alpha * (log_prob + target_entropy))
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha = tf.math.exp(self.log_alpha)
        else:
            self.alpha = 1.0
            alpha_loss = 0
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)

    def save(self):
        if False:
            print('Hello World!')
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.soft_q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.soft_q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_soft_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_soft_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        np.save(extend_path('log_alpha.npy'), self.log_alpha.numpy())

    def load_weights(self):
        if False:
            while True:
                i = 10
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        self.log_alpha.assign(np.load(extend_path('log_alpha.npy')))
if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = SAC(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)
    t0 = time.time()
    if args.train:
        frame_idx = 0
        all_episode_reward = []
        state = env.reset().astype(np.float32)
        agent.policy_net([state])
        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state)
                else:
                    action = agent.policy_net.sample_action()
                (next_state, reward, done, _) = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done is True else 0
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                frame_idx += 1
                if len(replay_buffer) > BATCH_SIZE:
                    for i in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE, reward_scale=REWARD_SCALE, auto_entropy=AUTO_ENTROPY, target_entropy=-1.0 * action_dim)
                if done:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
    if args.test:
        agent.load_weights()
        state = env.reset().astype(np.float32)
        agent.policy_net([state])
        for episode in range(TEST_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                (state, reward, done, info) = env.step(agent.policy_net.get_action(state, greedy=True))
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))