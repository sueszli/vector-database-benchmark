from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
    if False:
        while True:
            i = 10
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:

    def __init__(self, M):
        if False:
            print('Hello World!')
        self.M = M

    def fit(self, X, max_iter=30):
        if False:
            return 10
        np.random.seed(123)
        V = max((max(x) for x in X)) + 1
        N = len(X)
        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, V)
        print('initial A:', self.A)
        print('initial B:', self.B)
        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print('it:', it)
            alphas = []
            betas = []
            scales = []
            logP = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                scale = np.zeros(T)
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi * self.B[:, x[0]]
                scale[0] = alpha[0].sum()
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha_t_prime = alpha[t - 1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                logP[n] = np.log(scale).sum()
                alphas.append(alpha)
                scales.append(scale)
                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t + 1]] * beta[t + 1]) / scale[t + 1]
                betas.append(beta)
            cost = np.sum(logP)
            costs.append(cost)
            self.pi = np.sum((alphas[n][0] * betas[n][0] for n in range(N))) / N
            den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = np.zeros((self.M, self.M))
            b_num = np.zeros((self.M, V))
            for n in range(N):
                x = X[n]
                T = len(x)
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T - 1):
                            a_num[i, j] += alphas[n][t, i] * betas[n][t + 1, j] * self.A[i, j] * self.B[j, x[t + 1]] / scales[n][t + 1]
                for i in range(self.M):
                    for t in range(T):
                        b_num[i, x[t]] += alphas[n][t, i] * betas[n][t, i]
            self.A = a_num / den1
            self.B = b_num / den2
        print('A:', self.A)
        print('B:', self.B)
        print('pi:', self.pi)
        plt.plot(costs)
        plt.show()

    def log_likelihood(self, x):
        if False:
            i = 10
            return i + 15
        T = len(x)
        scale = np.zeros(T)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t - 1].dot(self.A) * self.B[:, x[t]]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        if False:
            return 10
        return np.array([self.log_likelihood(x) for x in X])

    def get_state_sequence(self, x):
        if False:
            for i in range(10):
                print('nop')
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(self.B[:, x[0]])
        for t in range(1, T):
            for j in range(self.M):
                delta[t, j] = np.max(delta[t - 1] + np.log(self.A[:, j])) + np.log(self.B[j, x[t]])
                psi[t, j] = np.argmax(delta[t - 1] + np.log(self.A[:, j]))
        states = np.zeros(T, dtype=np.int32)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

def fit_coin():
    if False:
        while True:
            i = 10
    X = []
    for line in open('coin_data.txt'):
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)
    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.log_likelihood_multi(X).sum()
    print('LL with fitted params:', L)
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmm.log_likelihood_multi(X).sum()
    print('LL with true params:', L)
    print('Best state sequence for:', X[0])
    print(hmm.get_state_sequence(X[0]))
if __name__ == '__main__':
    fit_coin()