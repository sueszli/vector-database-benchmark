from __future__ import print_function, division
from builtins import range
import wave
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from generate_c import get_signals, big_init

class HMM:

    def __init__(self, M, K):
        if False:
            for i in range(10):
                print('nop')
        self.M = M
        self.K = K

    def fit(self, X, learning_rate=0.01, max_iter=10):
        if False:
            print('Hello World!')
        N = len(X)
        D = X[0].shape[1]
        pi0 = np.ones(self.M)
        A0 = np.random.randn(self.M, self.M)
        R0 = np.ones((self.M, self.K))
        mu0 = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                mu0[i, k] = x[random_time_idx]
        sigma0 = np.random.randn(self.M, self.K, D, D)
        (thx, cost) = self.set(pi0, A0, R0, mu0, sigma0)
        pi_update = self.preSoftmaxPi - learning_rate * T.grad(cost, self.preSoftmaxPi)
        A_update = self.preSoftmaxA - learning_rate * T.grad(cost, self.preSoftmaxA)
        R_update = self.preSoftmaxR - learning_rate * T.grad(cost, self.preSoftmaxR)
        mu_update = self.mu - learning_rate * T.grad(cost, self.mu)
        sigma_update = self.sigmaFactor - learning_rate * T.grad(cost, self.sigmaFactor)
        updates = [(self.preSoftmaxPi, pi_update), (self.preSoftmaxA, A_update), (self.preSoftmaxR, R_update), (self.mu, mu_update), (self.sigmaFactor, sigma_update)]
        train_op = theano.function(inputs=[thx], updates=updates)
        costs = []
        for it in range(max_iter):
            print('it:', it)
            for n in range(N):
                c = self.log_likelihood_multi(X).sum()
                print('c:', c)
                costs.append(c)
                train_op(X[n])
        plt.plot(costs)
        plt.show()

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, sigmaFactor):
        if False:
            print('Hello World!')
        self.preSoftmaxPi = theano.shared(preSoftmaxPi)
        self.preSoftmaxA = theano.shared(preSoftmaxA)
        self.preSoftmaxR = theano.shared(preSoftmaxR)
        self.mu = theano.shared(mu)
        self.sigmaFactor = theano.shared(sigmaFactor)
        (M, K) = preSoftmaxR.shape
        self.M = M
        self.K = K
        pi = T.nnet.softmax(self.preSoftmaxPi).flatten()
        A = T.nnet.softmax(self.preSoftmaxA)
        R = T.nnet.softmax(self.preSoftmaxR)
        D = self.mu.shape[2]
        twopiD = (2 * np.pi) ** D
        thx = T.matrix('X')

        def mvn_pdf(x, m, S):
            if False:
                return 10
            k = 1 / T.sqrt(twopiD * T.nlinalg.det(S))
            e = T.exp(-0.5 * (x - m).T.dot(T.nlinalg.matrix_inverse(S).dot(x - m)))
            return k * e

        def gmm_pdf(x):
            if False:
                return 10

            def state_pdfs(xt):
                if False:
                    while True:
                        i = 10

                def component_pdf(j, xt):
                    if False:
                        while True:
                            i = 10
                    Bj_t = 0
                    for k in range(self.K):
                        L = self.sigmaFactor[j, k]
                        S = L.dot(L.T)
                        Bj_t += R[j, k] * mvn_pdf(xt, self.mu[j, k], S)
                    return Bj_t
                (Bt, _) = theano.scan(fn=component_pdf, sequences=T.arange(self.M), n_steps=self.M, outputs_info=None, non_sequences=[xt])
                return Bt
            (B, _) = theano.scan(fn=state_pdfs, sequences=x, n_steps=x.shape[0], outputs_info=None)
            return B.T
        B = gmm_pdf(thx)

        def recurrence(t, old_a, B):
            if False:
                for i in range(10):
                    print('nop')
            a = old_a.dot(A) * B[:, t]
            s = a.sum()
            return (a / s, s)
        ([alpha, scale], _) = theano.scan(fn=recurrence, sequences=T.arange(1, thx.shape[0]), outputs_info=[pi * B[:, 0], None], n_steps=thx.shape[0] - 1, non_sequences=[B])
        cost = -T.log(scale).sum()
        self.cost_op = theano.function(inputs=[thx], outputs=cost)
        return (thx, cost)

    def log_likelihood_multi(self, X):
        if False:
            while True:
                i = 10
        return np.array([self.cost_op(x) for x in X])

def real_signal():
    if False:
        return 10
    spf = wave.open('helloworld.wav', 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()
    hmm = HMM(3, 3)
    hmm.fit(signal.reshape(1, T, 1), learning_rate=2e-07, max_iter=20)

def fake_signal():
    if False:
        print('Hello World!')
    signals = get_signals()
    hmm = HMM(5, 3)
    hmm.fit(signals, max_iter=3)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for fitted params:', L)
    (_, _, _, pi, A, R, mu, sigma) = big_init()
    pi = np.log(pi)
    A = np.log(A)
    R = np.log(R)
    sigma = np.linalg.cholesky(sigma)
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_multi(signals).sum()
    print('LL for actual params:', L)
if __name__ == '__main__':
    fake_signal()