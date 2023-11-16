from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle

class HiddenLayer(object):

    def __init__(self, M1, M2, an_id):
        if False:
            return 10
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        if False:
            for i in range(10):
                print('nop')
        return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):

    def __init__(self, hidden_layer_sizes):
        if False:
            i = 10
            return i + 15
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=0.01, mu=0.99, reg=1e-12, epochs=400, batch_sz=20, print_period=1, show_fig=False):
        if False:
            for i in range(10):
                print('nop')
        Y = Y.astype(np.int32)
        (N, D) = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W = init_weight(M1, K)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        cache = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)
        rcost = reg * T.sum([(p * p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)
        updates = [(p, p + mu * dp - learning_rate * g) for (p, dp, g) in zip(self.params, dparams, grads)] + [(dp, mu * dp - learning_rate * g) for (dp, g) in zip(dparams, grads)]
        train_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction], updates=updates)
        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            (X, Y) = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:j * batch_sz + batch_sz]
                Ybatch = Y[j * batch_sz:j * batch_sz + batch_sz]
                (c, p) = train_op(Xbatch, Ybatch)
                if j % print_period == 0:
                    costs.append(c)
                    e = np.mean(Ybatch != p)
                    print('i:', i, 'j:', j, 'nb:', n_batches, 'cost:', c, 'error rate:', e)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        if False:
            for i in range(10):
                print('nop')
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        if False:
            return 10
        pY = self.forward(X)
        return T.argmax(pY, axis=1)

def wide():
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate=0.0001, print_period=10, epochs=300, show_fig=True)

def deep():
    if False:
        return 10
    (X, Y) = all_parity_pairs(12)
    model = ANN([1024] * 2)
    model.fit(X, Y, learning_rate=0.001, print_period=10, epochs=100, show_fig=True)
if __name__ == '__main__':
    wide()