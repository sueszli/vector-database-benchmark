import gym, os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from PIL import Image
env = gym.make('CartPole-v1')
env.seed(2333)
tf.random.set_seed(2333)
np.random.seed(2333)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
learning_rate = 0.0002
gamma = 0.98

class Policy(keras.Model):

    def __init__(self):
        if False:
            print('Hello World!')
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = layers.Dense(128, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')
        self.optimizer = optimizers.Adam(lr=learning_rate)

    def call(self, inputs, training=None):
        if False:
            print('Hello World!')
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.softmax(self.fc2(x), axis=1)
        return x

    def put_data(self, item):
        if False:
            print('Hello World!')
        self.data.append(item)

    def train_net(self, tape):
        if False:
            print('Hello World!')
        R = 0
        for (r, log_prob) in self.data[::-1]:
            R = r + gamma * R
            loss = -log_prob * R
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []

def main():
    if False:
        for i in range(10):
            print('nop')
    pi = Policy()
    pi(tf.random.normal((4, 4)))
    pi.summary()
    score = 0.0
    print_interval = 20
    returns = []
    for n_epi in range(400):
        s = env.reset()
        with tf.GradientTape(persistent=True) as tape:
            for t in range(501):
                s = tf.constant(s, dtype=tf.float32)
                s = tf.expand_dims(s, axis=0)
                prob = pi(s)
                a = tf.random.categorical(tf.math.log(prob), 1)[0]
                a = int(a)
                (s_prime, r, done, info) = env.step(a)
                pi.put_data((r, tf.math.log(prob[0][a])))
                s = s_prime
                score += r
                if n_epi > 1000:
                    env.render()
                if done:
                    break
            pi.train_net(tape)
        del tape
        if n_epi % print_interval == 0 and n_epi != 0:
            returns.append(score / print_interval)
            print(f'# of episode :{n_epi}, avg score : {score / print_interval}')
            score = 0.0
    env.close()
    plt.plot(np.arange(len(returns)) * print_interval, returns)
    plt.plot(np.arange(len(returns)) * print_interval, returns, 's')
    plt.xlabel('回合数')
    plt.ylabel('总回报')
    plt.savefig('reinforce-tf-cartpole.svg')
if __name__ == '__main__':
    main()