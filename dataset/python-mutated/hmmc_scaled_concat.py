from __future__ import print_function, division
from builtins import range
import wave
import numpy as np
import matplotlib.pyplot as plt
from generate_c import get_signals, big_init, simple_init
from scipy.stats import multivariate_normal as mvn

def random_normalized(d1, d2):
    if False:
        print('Hello World!')
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:

    def __init__(self, M, K):
        if False:
            return 10
        self.M = M
        self.K = K

    def fit(self, X, max_iter=25, eps=0.1):
        if False:
            print('Hello World!')
        sequenceLengths = []
        for x in X:
            sequenceLengths.append(len(x))
        Xc = np.concatenate(X)
        T = len(Xc)
        startPositions = np.zeros(len(Xc), dtype=np.bool)
        endPositions = np.zeros(len(Xc), dtype=np.bool)
        startPositionValues = []
        last = 0
        for length in sequenceLengths:
            startPositionValues.append(last)
            startPositions[last] = 1
            if last > 0:
                endPositions[last - 1] = 1
            last += length
        D = X[0].shape[1]
        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.R = np.ones((self.M, self.K)) / self.K
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(T)
                self.mu[i, k] = Xc[random_idx]
        self.sigma = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j, k] = np.eye(D)
        costs = []
        for it in range(max_iter):
            if it % 1 == 0:
                print('it:', it)
            scale = np.zeros(T)
            B = np.zeros((self.M, T))
            component = np.zeros((self.M, self.K, T))
            for j in range(self.M):
                for k in range(self.K):
                    p = self.R[j, k] * mvn.pdf(Xc, self.mu[j, k], self.sigma[j, k])
                    component[j, k, :] = p
                    B[j, :] += p
            alpha = np.zeros((T, self.M))
            alpha[0] = self.pi * B[:, 0]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, T):
                if startPositions[t] == 0:
                    alpha_t_prime = alpha[t - 1].dot(self.A) * B[:, t]
                else:
                    alpha_t_prime = self.pi * B[:, t]
                scale[t] = alpha_t_prime.sum()
                alpha[t] = alpha_t_prime / scale[t]
            logP = np.log(scale).sum()
            beta = np.zeros((T, self.M))
            beta[-1] = 1
            for t in range(T - 2, -1, -1):
                if startPositions[t + 1] == 1:
                    beta[t] = 1
                else:
                    beta[t] = self.A.dot(B[:, t + 1] * beta[t + 1]) / scale[t + 1]
            gamma = np.zeros((T, self.M, self.K))
            for t in range(T):
                alphabeta = alpha[t, :].dot(beta[t, :])
                for j in range(self.M):
                    factor = alpha[t, j] * beta[t, j] / alphabeta
                    for k in range(self.K):
                        gamma[t, j, k] = factor * component[j, k, t] / B[j, t]
            costs.append(logP)
            self.pi = np.sum((alpha[t] * beta[t] for t in startPositionValues)) / len(startPositionValues)
            a_den = np.zeros((self.M, 1))
            a_num = np.zeros((self.M, self.M))
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
            nonEndPositions = (1 - endPositions).astype(np.bool)
            a_den += (alpha[nonEndPositions] * beta[nonEndPositions]).sum(axis=0, keepdims=True).T
            for i in range(self.M):
                for j in range(self.M):
                    for t in range(T - 1):
                        if endPositions[t] != 1:
                            a_num[i, j] += alpha[t, i] * beta[t + 1, j] * self.A[i, j] * B[j, t + 1] / scale[t + 1]
            self.A = a_num / a_den
            r_num_n = np.zeros((self.M, self.K))
            r_den_n = np.zeros(self.M)
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        r_num_n[j, k] += gamma[t, j, k]
                        r_den_n[j] += gamma[t, j, k]
            r_num = r_num_n
            r_den = r_den_n
            mu_num_n = np.zeros((self.M, self.K, D))
            sigma_num_n = np.zeros((self.M, self.K, D, D))
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(T):
                        mu_num_n[j, k] += gamma[t, j, k] * Xc[t]
                        sigma_num_n[j, k] += gamma[t, j, k] * np.outer(Xc[t] - self.mu[j, k], Xc[t] - self.mu[j, k])
            mu_num = mu_num_n
            sigma_num = sigma_num_n
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j, k] = r_num[j, k] / r_den[j]
                    self.mu[j, k] = mu_num[j, k] / r_num[j, k]
                    self.sigma[j, k] = sigma_num[j, k] / r_num[j, k] + np.eye(D) * eps
            assert np.all(self.R <= 1)
            assert np.all(self.A <= 1)
        print('A:', self.A)
        print('mu:', self.mu)
        print('sigma:', self.sigma)
        print('R:', self.R)
        print('pi:', self.pi)
        plt.plot(costs)
        plt.show()

    def log_likelihood(self, x):
        if False:
            for i in range(10):
                print('nop')
        T = len(x)
        scale = np.zeros(T)
        B = np.zeros((self.M, T))
        for j in range(self.M):
            for k in range(self.K):
                p = self.R[j, k] * mvn.pdf(x, self.mu[j, k], self.sigma[j, k])
                B[j, :] += p
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * B[:, 0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t - 1].dot(self.A) * B[:, t]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        if False:
            i = 10
            return i + 15
        return np.array([self.log_likelihood(x) for x in X])

    def set(self, pi, A, R, mu, sigma):
        if False:
            i = 10
            return i + 15
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        (M, K) = R.shape
        self.M = M
        self.K = K

def real_signal():
    if False:
        print('Hello World!')
    spf = wave.open('helloworld.wav', 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()
    hmm = HMM(5, 3)
    hmm.fit(signal.reshape(1, T, 1))
    print('LL for fitted params:', hmm.log_likelihood(signal.reshape(T, 1)))

def fake_signal(init=big_init):
    if False:
        for i in range(10):
            print('nop')
    signals = get_signals(init=init)
    hmm = HMM(5, 3)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for fitted params:', L)
    (_, _, _, pi, A, R, mu, sigma) = init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for actual params:', L)
if __name__ == '__main__':
    fake_signal()