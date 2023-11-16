from __future__ import print_function, division
from builtins import range
import wave
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
MVN = tf.contrib.distributions.MultivariateNormalDiag
from generate_c import get_signals, big_init

class HMM:

    def __init__(self, M, K, D):
        if False:
            while True:
                i = 10
        self.M = M
        self.K = K
        self.D = D

    def set_session(self, session):
        if False:
            i = 10
            return i + 15
        self.session = session

    def init_random(self, X):
        if False:
            print('Hello World!')
        pi0 = np.ones(self.M).astype(np.float32)
        A0 = np.random.randn(self.M, self.M).astype(np.float32)
        R0 = np.ones((self.M, self.K)).astype(np.float32)
        mu0 = np.zeros((self.M, self.K, self.D))
        for j in range(self.M):
            for k in range(self.K):
                n = np.random.randint(X.shape[0])
                t = np.random.randint(X.shape[1])
                mu0[j, k] = X[n, t]
        mu0 = mu0.astype(np.float32)
        sigma0 = np.random.randn(self.M, self.K, self.D).astype(np.float32)
        self.build(pi0, A0, R0, mu0, sigma0)

    def build(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, logSigma):
        if False:
            for i in range(10):
                print('nop')
        self.preSoftmaxPi = tf.Variable(preSoftmaxPi)
        self.preSoftmaxA = tf.Variable(preSoftmaxA)
        self.preSoftmaxR = tf.Variable(preSoftmaxR)
        self.mu = tf.Variable(mu)
        self.logSigma = tf.Variable(logSigma)
        pi = tf.nn.softmax(self.preSoftmaxPi)
        A = tf.nn.softmax(self.preSoftmaxA)
        R = tf.nn.softmax(self.preSoftmaxR)
        sigma = tf.exp(self.logSigma)
        self.tfx = tf.placeholder(tf.float32, shape=(None, self.D), name='X')
        self.mvns = []
        for j in range(self.M):
            self.mvns.append([])
            for k in range(self.K):
                self.mvns[j].append(MVN(self.mu[j, k], sigma[j, k]))
        B = []
        for j in range(self.M):
            components = []
            for k in range(self.K):
                components.append(self.mvns[j][k].prob(self.tfx))
            components = tf.stack(components)
            R_j = tf.reshape(R[j], [1, self.K])
            p_x_t_j = tf.matmul(R_j, components)
            components = tf.reshape(p_x_t_j, [-1])
            B.append(components)
        B = tf.stack(B)
        B = tf.transpose(B, [1, 0])

        def recurrence(old_a_old_s, B_t):
            if False:
                while True:
                    i = 10
            old_a = tf.reshape(old_a_old_s[0], (1, self.M))
            a = tf.matmul(old_a, A) * B_t
            a = tf.reshape(a, (self.M,))
            s = tf.reduce_sum(a)
            return (a / s, s)
        (alpha, scale) = tf.scan(fn=recurrence, elems=B[1:], initializer=(pi * B[0], np.float32(1.0)))
        self.cost_op = -tf.reduce_sum(tf.log(scale))
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.cost_op)

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, logSigma):
        if False:
            print('Hello World!')
        op1 = self.preSoftmaxPi.assign(preSoftmaxPi)
        op2 = self.preSoftmaxA.assign(preSoftmaxA)
        op3 = self.preSoftmaxR.assign(preSoftmaxR)
        op4 = self.mu.assign(mu)
        op5 = self.logSigma.assign(logSigma)
        self.session.run([op1, op2, op3, op4, op5])

    def fit(self, X, max_iter=10):
        if False:
            for i in range(10):
                print('nop')
        N = len(X)
        print('number of train samples:', N)
        costs = []
        for it in range(max_iter):
            if it % 1 == 0:
                print('it:', it)
            for n in range(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                self.session.run(self.train_op, feed_dict={self.tfx: X[n]})
        plt.plot(costs)
        plt.show()

    def get_cost(self, x):
        if False:
            print('Hello World!')
        return self.session.run(self.cost_op, feed_dict={self.tfx: x})

    def get_cost_multi(self, X):
        if False:
            i = 10
            return i + 15
        return np.array([self.get_cost(x) for x in X])

def real_signal():
    if False:
        while True:
            i = 10
    spf = wave.open('helloworld.wav', 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()
    signals = signal.reshape(1, T, 1)
    hmm = HMM(3, 3, 1)
    hmm.init_random(signals)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    hmm.set_session(session)
    hmm.fit(signals, max_iter=30)

def fake_signal():
    if False:
        for i in range(10):
            print('nop')
    signals = get_signals()
    signals = np.array(signals)
    hmm = HMM(5, 3, signals[0].shape[1])
    hmm.init_random(signals)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    hmm.set_session(session)
    hmm.fit(signals, max_iter=30)
    L = hmm.get_cost_multi(signals).sum()
    print('LL for fitted params:', L)
    (_, _, _, pi, A, R, mu, sigma) = big_init()
    pi = np.log(pi)
    A = np.log(A)
    R = np.log(R)
    (M, K, D, _) = sigma.shape
    logSigma = np.zeros((M, K, D))
    for j in range(M):
        for k in range(D):
            logSigma[j, k] = np.log(np.diag(sigma[j, k]))
    hmm.set(pi, A, R, mu, logSigma)
    L = hmm.get_cost_multi(signals).sum()
    print('LL for actual params:', L)
if __name__ == '__main__':
    fake_signal()