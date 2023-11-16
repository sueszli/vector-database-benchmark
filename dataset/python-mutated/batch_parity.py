from __future__ import print_function, division
from builtins import range
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels

class SimpleRNN:

    def __init__(self, M):
        if False:
            while True:
                i = 10
        self.M = M

    def fit(self, X, Y, batch_sz=20, learning_rate=1.0, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        if False:
            i = 10
            return i + 15
        D = X[0].shape[1]
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        thStartPoints = T.ivector('start_points')
        XW = thX.dot(self.Wx)

        def recurrence(xw_t, is_start, h_t1, h0):
            if False:
                print('Hello World!')
            h_t = T.switch(T.eq(is_start, 1), self.f(xw_t + h0.dot(self.Wh) + self.bh), self.f(xw_t + h_t1.dot(self.Wh) + self.bh))
            return h_t
        (h, _) = theano.scan(fn=recurrence, outputs_info=[self.h0], sequences=[XW, thStartPoints], non_sequences=[self.h0], n_steps=XW.shape[0])
        py_x = T.nnet.softmax(h.dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]
        updates = [(p, p + mu * dp - learning_rate * g) for (p, dp, g) in zip(self.params, dparams, grads)] + [(dp, mu * dp - learning_rate * g) for (dp, g) in zip(dparams, grads)]
        self.train_op = theano.function(inputs=[thX, thY, thStartPoints], outputs=[cost, prediction, py_x], updates=updates)
        costs = []
        n_batches = N // batch_sz
        sequenceLength = X.shape[1]
        startPoints = np.zeros(sequenceLength * batch_sz, dtype=np.int32)
        for b in range(batch_sz):
            startPoints[b * sequenceLength] = 1
        for i in range(epochs):
            (X, Y) = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(n_batches):
                Xbatch = X[j * batch_sz:(j + 1) * batch_sz].reshape(sequenceLength * batch_sz, D)
                Ybatch = Y[j * batch_sz:(j + 1) * batch_sz].reshape(sequenceLength * batch_sz).astype(np.int32)
                (c, p, rout) = self.train_op(Xbatch, Ybatch, startPoints)
                cost += c
                for b in range(batch_sz):
                    idx = sequenceLength * (b + 1) - 1
                    if p[idx] == Ybatch[idx]:
                        n_correct += 1
            if i % 10 == 0:
                print('shape y:', rout.shape)
                print('i:', i, 'cost:', cost, 'classification rate:', float(n_correct) / N)
            if n_correct == N:
                print('i:', i, 'cost:', cost, 'classification rate:', float(n_correct) / N)
                break
            costs.append(cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

def parity(B=12, learning_rate=0.001, epochs=3000):
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = all_parity_pairs_with_sequence_labels(B)
    rnn = SimpleRNN(4)
    rnn.fit(X, Y, batch_sz=10, learning_rate=learning_rate, epochs=epochs, activation=T.nnet.sigmoid, show_fig=False)
if __name__ == '__main__':
    parity()