from __future__ import print_function, division
from builtins import range
import wave
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from generate_c import get_signals, big_init

def random_normalized(d1, d2):
    if False:
        print('Hello World!')
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:

    def __init__(self, M, K):
        if False:
            for i in range(10):
                print('nop')
        self.M = M
        self.K = K

    def fit(self, X, learning_rate=0.01, max_iter=10):
        if False:
            while True:
                i = 10
        N = len(X)
        D = X[0].shape[1]
        pi0 = np.ones(self.M) / self.M
        A0 = random_normalized(self.M, self.M)
        R0 = np.ones((self.M, self.K)) / self.K
        mu0 = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                mu0[i, k] = x[random_time_idx]
        sigma0 = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                sigma0[j, k] = np.eye(D)
        (thx, cost) = self.set(pi0, A0, R0, mu0, sigma0)
        pi_update = self.pi - learning_rate * T.grad(cost, self.pi)
        pi_update = pi_update / pi_update.sum()
        A_update = self.A - learning_rate * T.grad(cost, self.A)
        A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')
        R_update = self.R - learning_rate * T.grad(cost, self.R)
        R_update = R_update / R_update.sum(axis=1).dimshuffle(0, 'x')
        updates = [(self.pi, pi_update), (self.A, A_update), (self.R, R_update), (self.mu, self.mu - learning_rate * T.grad(cost, self.mu)), (self.sigma, self.sigma - learning_rate * T.grad(cost, self.sigma))]
        train_op = theano.function(inputs=[thx], updates=updates)
        costs = []
        for it in range(max_iter):
            print('it:', it)
            for n in range(N):
                c = self.log_likelihood_multi(X).sum()
                print('c:', c)
                costs.append(c)
                train_op(X[n])
        print('A:', self.A.get_value())
        print('mu:', self.mu.get_value())
        print('sigma:', self.sigma.get_value())
        print('R:', self.R.get_value())
        print('pi:', self.pi.get_value())
        plt.plot(costs)
        plt.show()

    def set(self, pi, A, R, mu, sigma):
        if False:
            return 10
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.R = theano.shared(R)
        self.mu = theano.shared(mu)
        self.sigma = theano.shared(sigma)
        (M, K) = R.shape
        self.M = M
        self.K = K
        D = self.mu.shape[2]
        twopiD = (2 * np.pi) ** D
        thx = T.matrix('X')

        def mvn_pdf(x, mu, sigma):
            if False:
                i = 10
                return i + 15
            k = 1 / T.sqrt(twopiD * T.nlinalg.det(sigma))
            e = T.exp(-0.5 * (x - mu).T.dot(T.nlinalg.matrix_inverse(sigma).dot(x - mu)))
            return k * e

        def gmm_pdf(x):
            if False:
                i = 10
                return i + 15

            def state_pdfs(xt):
                if False:
                    return 10

                def component_pdf(j, xt):
                    if False:
                        for i in range(10):
                            print('nop')
                    Bj_t = 0
                    for k in range(self.K):
                        Bj_t += self.R[j, k] * mvn_pdf(xt, self.mu[j, k], self.sigma[j, k])
                    return Bj_t
                (Bt, _) = theano.scan(fn=component_pdf, sequences=T.arange(self.M), n_steps=self.M, outputs_info=None, non_sequences=[xt])
                return Bt
            (B, _) = theano.scan(fn=state_pdfs, sequences=x, n_steps=x.shape[0], outputs_info=None)
            return B.T
        B = gmm_pdf(thx)

        def recurrence(t, old_a, B):
            if False:
                while True:
                    i = 10
            a = old_a.dot(self.A) * B[:, t]
            s = a.sum()
            return (a / s, s)
        ([alpha, scale], _) = theano.scan(fn=recurrence, sequences=T.arange(1, thx.shape[0]), outputs_info=[self.pi * B[:, 0], None], n_steps=thx.shape[0] - 1, non_sequences=[B])
        cost = -T.log(scale).sum()
        self.cost_op = theano.function(inputs=[thx], outputs=cost)
        return (thx, cost)

    def log_likelihood_multi(self, X):
        if False:
            print('Hello World!')
        return np.array([self.cost_op(x) for x in X])

def real_signal():
    if False:
        for i in range(10):
            print('nop')
    spf = wave.open('helloworld.wav', 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()
    hmm = HMM(5, 3)
    hmm.fit(signal.reshape(1, T, 1), learning_rate=1e-05, max_iter=20)

def fake_signal():
    if False:
        i = 10
        return i + 15
    signals = get_signals()
    hmm = HMM(5, 3)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for fitted params:', L)
    (_, _, _, pi, A, R, mu, sigma) = big_init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for actual params:', L)
if __name__ == '__main__':
    fake_signal()