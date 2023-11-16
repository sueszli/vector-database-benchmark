from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def dot1(V, W):
    if False:
        while True:
            i = 10
    return tf.tensordot(V, W, axes=[[1, 2], [0, 1]])

def dot2(H, W):
    if False:
        print('Hello World!')
    return tf.tensordot(H, W, axes=[[1], [2]])

class RBM(object):

    def __init__(self, D, M, K):
        if False:
            print('Hello World!')
        self.D = D
        self.M = M
        self.K = K
        self.build(D, M, K)

    def build(self, D, M, K):
        if False:
            return 10
        self.W = tf.Variable(tf.random.normal(shape=(D, K, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))
        self.X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
        X = tf.cast(self.X_in * 2 - 1, tf.int32)
        X = tf.one_hot(X, K)
        V = X
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v
        r = tf.random.uniform(shape=tf.shape(input=p_h_given_v))
        H = tf.cast(r < p_h_given_v, dtype=tf.float32)
        logits = dot2(H, self.W) + self.b
        cdist = tf.compat.v1.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()
        X_sample = tf.one_hot(X_sample, depth=K)
        mask2d = tf.cast(self.X_in > 0, tf.float32)
        mask3d = tf.stack([mask2d] * K, axis=-1)
        X_sample = X_sample * mask3d
        objective = tf.reduce_mean(input_tensor=self.free_energy(X)) - tf.reduce_mean(input_tensor=self.free_energy(X_sample))
        self.train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(objective)
        logits = self.forward_logits(X)
        self.cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(X), logits=logits))
        self.output_visible = self.forward_output(X)
        self.one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32) / 2)
        self.pred = tf.tensordot(self.output_visible, self.one_to_ten, axes=[[2], [0]])
        mask = tf.cast(self.X_in > 0, tf.float32)
        se = mask * (self.X_in - self.pred) * (self.X_in - self.pred)
        self.sse = tf.reduce_sum(input_tensor=se)
        self.X_test = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
        mask = tf.cast(self.X_test > 0, tf.float32)
        tse = mask * (self.X_test - self.pred) * (self.X_test - self.pred)
        self.tsse = tf.reduce_sum(input_tensor=tse)
        initop = tf.compat.v1.global_variables_initializer()
        self.session = tf.compat.v1.Session()
        self.session.run(initop)

    def fit(self, X, X_test, epochs=10, batch_sz=256, show_fig=True):
        if False:
            i = 10
            return i + 15
        (N, D) = X.shape
        n_batches = N // batch_sz
        costs = []
        test_costs = []
        for i in range(epochs):
            t0 = datetime.now()
            print('epoch:', i)
            (X, X_test) = shuffle(X, X_test)
            for j in range(n_batches):
                x = X[j * batch_sz:j * batch_sz + batch_sz].toarray()
                (_, c) = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: x})
                if j % 100 == 0:
                    print('j / n_batches:', j, '/', n_batches, 'cost:', c)
            print('duration:', datetime.now() - t0)
            t0 = datetime.now()
            sse = 0
            test_sse = 0
            n = 0
            test_n = 0
            for j in range(n_batches):
                x = X[j * batch_sz:j * batch_sz + batch_sz].toarray()
                xt = X_test[j * batch_sz:j * batch_sz + batch_sz].toarray()
                n += np.count_nonzero(x)
                test_n += np.count_nonzero(xt)
                (sse_j, tsse_j) = self.get_sse(x, xt)
                sse += sse_j
                test_sse += tsse_j
            c = sse / n
            ct = test_sse / test_n
            print('train mse:', c)
            print('test mse:', ct)
            print('calculate cost duration:', datetime.now() - t0)
            costs.append(c)
            test_costs.append(ct)
        if show_fig:
            plt.plot(costs, label='train mse')
            plt.plot(test_costs, label='test mse')
            plt.legend()
            plt.show()

    def free_energy(self, V):
        if False:
            for i in range(10):
                print('nop')
        first_term = -tf.reduce_sum(input_tensor=dot1(V, self.b))
        second_term = -tf.reduce_sum(input_tensor=tf.nn.softplus(dot1(V, self.W) + self.c), axis=1)
        return first_term + second_term

    def forward_hidden(self, X):
        if False:
            print('Hello World!')
        return tf.nn.sigmoid(dot1(X, self.W) + self.c)

    def forward_logits(self, X):
        if False:
            i = 10
            return i + 15
        Z = self.forward_hidden(X)
        return dot2(Z, self.W) + self.b

    def forward_output(self, X):
        if False:
            i = 10
            return i + 15
        return tf.nn.softmax(self.forward_logits(X))

    def transform(self, X):
        if False:
            print('Hello World!')
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

    def get_visible(self, X):
        if False:
            for i in range(10):
                print('nop')
        return self.session.run(self.output_visible, feed_dict={self.X_in: X})

    def get_sse(self, X, Xt):
        if False:
            while True:
                i = 10
        return self.session.run((self.sse, self.tsse), feed_dict={self.X_in: X, self.X_test: Xt})

def main():
    if False:
        while True:
            i = 10
    A = load_npz('Atrain.npz')
    A_test = load_npz('Atest.npz')
    (N, M) = A.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A, A_test)
if __name__ == '__main__':
    main()