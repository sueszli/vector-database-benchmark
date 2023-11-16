from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class HMM:

    def __init__(self, M):
        if False:
            return 10
        self.M = M

    def fit(self, X, learning_rate=0.001, max_iter=10, V=None, print_period=1):
        if False:
            i = 10
            return i + 15
        if V is None:
            V = max((max(x) for x in X)) + 1
        N = len(X)
        print('number of train samples:', N)
        preSoftmaxPi0 = np.zeros(self.M)
        preSoftmaxA0 = np.random.randn(self.M, self.M)
        preSoftmaxB0 = np.random.randn(self.M, V)
        (thx, cost) = self.set(preSoftmaxPi0, preSoftmaxA0, preSoftmaxB0)
        pi_update = self.preSoftmaxPi - learning_rate * T.grad(cost, self.preSoftmaxPi)
        A_update = self.preSoftmaxA - learning_rate * T.grad(cost, self.preSoftmaxA)
        B_update = self.preSoftmaxB - learning_rate * T.grad(cost, self.preSoftmaxB)
        updates = [(self.preSoftmaxPi, pi_update), (self.preSoftmaxA, A_update), (self.preSoftmaxB, B_update)]
        train_op = theano.function(inputs=[thx], updates=updates, allow_input_downcast=True)
        costs = []
        for it in range(max_iter):
            if it % print_period == 0:
                print('it:', it)
            for n in range(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                train_op(X[n])
        plt.plot(costs)
        plt.show()

    def get_cost(self, x):
        if False:
            print('Hello World!')
        return self.cost_op(x)

    def log_likelihood(self, x):
        if False:
            print('Hello World!')
        return -self.cost_op(x)

    def get_cost_multi(self, X):
        if False:
            return 10
        return np.array([self.get_cost(x) for x in X])

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        if False:
            while True:
                i = 10
        self.preSoftmaxPi = theano.shared(preSoftmaxPi)
        self.preSoftmaxA = theano.shared(preSoftmaxA)
        self.preSoftmaxB = theano.shared(preSoftmaxB)
        pi = T.nnet.softmax(self.preSoftmaxPi).flatten()
        A = T.nnet.softmax(self.preSoftmaxA)
        B = T.nnet.softmax(self.preSoftmaxB)
        thx = T.ivector('thx')

        def recurrence(t, old_a, x):
            if False:
                while True:
                    i = 10
            a = old_a.dot(A) * B[:, x[t]]
            s = a.sum()
            return (a / s, s)
        ([alpha, scale], _) = theano.scan(fn=recurrence, sequences=T.arange(1, thx.shape[0]), outputs_info=[pi * B[:, thx[0]], None], n_steps=thx.shape[0] - 1, non_sequences=thx)
        cost = -T.log(scale).sum()
        self.cost_op = theano.function(inputs=[thx], outputs=cost, allow_input_downcast=True)
        return (thx, cost)

def fit_coin():
    if False:
        print('Hello World!')
    X = []
    for line in open('coin_data.txt'):
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)
    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.get_cost_multi(X).sum()
    print('LL with fitted params:', L)
    pi = np.log(np.array([0.5, 0.5]))
    A = np.log(np.array([[0.1, 0.9], [0.8, 0.2]]))
    B = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
    hmm.set(pi, A, B)
    L = hmm.get_cost_multi(X).sum()
    print('LL with true params:', L)
if __name__ == '__main__':
    fit_coin()