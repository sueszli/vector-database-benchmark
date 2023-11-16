from __future__ import print_function, division
from builtins import range
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from util import get_normalized_data
from sklearn.utils import shuffle

def momentum_updates(cost, params, lr, mu):
    if False:
        for i in range(10):
            print('nop')
    grads = T.grad(cost, params)
    updates = []
    for (p, g) in zip(params, grads):
        dp = theano.shared(p.get_value() * 0)
        new_dp = mu * dp - lr * g
        new_p = p + new_dp
        updates.append((dp, new_dp))
        updates.append((p, new_p))
    return updates

class HiddenLayer(object):

    def __init__(self, M1, M2, an_id):
        if False:
            for i in range(10):
                print('nop')
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        if False:
            print('Hello World!')
        return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):

    def __init__(self, hidden_layer_sizes, p_keep):
        if False:
            i = 10
            return i + 15
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=0.01, mu=0.9, decay=0.9, epochs=10, batch_sz=100, show_fig=False):
        if False:
            i = 10
            return i + 15
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int32)
        self.rng = RandomStreams()
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
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY_train = self.forward_train(thX)
        cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))
        updates = momentum_updates(cost, self.params, learning_rate, mu)
        train_op = theano.function(inputs=[thX, thY], updates=updates)
        pY_predict = self.forward_predict(thX)
        cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost_predict, prediction])
        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            (X, Y) = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:j * batch_sz + batch_sz]
                Ybatch = Y[j * batch_sz:j * batch_sz + batch_sz]
                train_op(Xbatch, Ybatch)
                if j % 50 == 0:
                    (c, p) = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print('i:', i, 'j:', j, 'nb:', n_batches, 'cost:', c, 'error rate:', e)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_train(self, X):
        if False:
            print('Hello World!')
        Z = X
        for (h, p) in zip(self.hidden_layers, self.dropout_rates[:-1]):
            mask = self.rng.binomial(n=1, p=p, size=Z.shape)
            Z = mask * Z
            Z = h.forward(Z)
        mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
        Z = mask * Z
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def forward_predict(self, X):
        if False:
            print('Hello World!')
        Z = X
        for (h, p) in zip(self.hidden_layers, self.dropout_rates[:-1]):
            Z = h.forward(p * Z)
        return T.nnet.softmax((self.dropout_rates[-1] * Z).dot(self.W) + self.b)

    def predict(self, X):
        if False:
            print('Hello World!')
        pY = self.forward_predict(X)
        return T.argmax(pY, axis=1)

def error_rate(p, t):
    if False:
        i = 10
        return i + 15
    return np.mean(p != t)

def relu(a):
    if False:
        print('Hello World!')
    return a * (a > 0)

def main():
    if False:
        while True:
            i = 10
    (Xtrain, Xtest, Ytrain, Ytest) = get_normalized_data()
    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(Xtrain, Ytrain, Xtest, Ytest, show_fig=True)
if __name__ == '__main__':
    main()