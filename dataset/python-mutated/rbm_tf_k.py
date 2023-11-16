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

def one_hot_encode(X, K):
    if False:
        print('Hello World!')
    (N, D) = X.shape
    Y = np.zeros((N, D, K))
    for (n, d) in zip(*X.nonzero()):
        k = int(X[n, d] * 2 - 1)
        Y[n, d, k] = 1
    return Y

def one_hot_mask(X, K):
    if False:
        for i in range(10):
            print('nop')
    (N, D) = X.shape
    Y = np.zeros((N, D, K))
    for (n, d) in zip(*X.nonzero()):
        Y[n, d, :] = 1
    return Y
one_to_ten = np.arange(10) + 1

def convert_probs_to_ratings(probs):
    if False:
        i = 10
        return i + 15
    return probs.dot(one_to_ten) / 2

def dot1(V, W):
    if False:
        return 10
    return tf.tensordot(V, W, axes=[[1, 2], [0, 1]])

def dot2(H, W):
    if False:
        print('Hello World!')
    return tf.tensordot(H, W, axes=[[1], [2]])

class RBM(object):

    def __init__(self, D, M, K):
        if False:
            while True:
                i = 10
        self.D = D
        self.M = M
        self.K = K
        self.build(D, M, K)

    def build(self, D, M, K):
        if False:
            while True:
                i = 10
        self.W = tf.Variable(tf.random.normal(shape=(D, K, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))
        self.X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, D, K))
        self.mask = tf.compat.v1.placeholder(tf.float32, shape=(None, D, K))
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v
        r = tf.random.uniform(shape=tf.shape(input=p_h_given_v))
        H = tf.cast(r < p_h_given_v, dtype=tf.float32)
        logits = dot2(H, self.W) + self.b
        cdist = tf.compat.v1.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()
        X_sample = tf.one_hot(X_sample, depth=K)
        X_sample = X_sample * self.mask
        objective = tf.reduce_mean(input_tensor=self.free_energy(self.X_in)) - tf.reduce_mean(input_tensor=self.free_energy(X_sample))
        self.train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(objective)
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(self.X_in), logits=logits))
        self.output_visible = self.forward_output(self.X_in)
        initop = tf.compat.v1.global_variables_initializer()
        self.session = tf.compat.v1.Session()
        self.session.run(initop)

    def fit(self, X, mask, X_test, mask_test, epochs=10, batch_sz=256, show_fig=True):
        if False:
            while True:
                i = 10
        (N, D) = X.shape
        n_batches = N // batch_sz
        costs = []
        test_costs = []
        for i in range(epochs):
            t0 = datetime.now()
            print('epoch:', i)
            (X, mask, X_test, mask_test) = shuffle(X, mask, X_test, mask_test)
            for j in range(n_batches):
                x = X[j * batch_sz:j * batch_sz + batch_sz].toarray()
                m = mask[j * batch_sz:j * batch_sz + batch_sz].toarray()
                batch_one_hot = one_hot_encode(x, self.K)
                m = one_hot_mask(m, self.K)
                (_, c) = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch_one_hot, self.mask: m})
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
                m = mask[j * batch_sz:j * batch_sz + batch_sz].toarray()
                xoh = one_hot_encode(x, self.K)
                probs = self.get_visible(xoh)
                xhat = convert_probs_to_ratings(probs)
                sse += (m * (xhat - x) * (xhat - x)).sum()
                n += m.sum()
                xt = X_test[j * batch_sz:j * batch_sz + batch_sz].toarray()
                mt = mask_test[j * batch_sz:j * batch_sz + batch_sz].toarray()
                test_sse += (mt * (xhat - xt) * (xhat - xt)).sum()
                test_n += mt.sum()
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
            while True:
                i = 10
        first_term = -tf.reduce_sum(input_tensor=dot1(V, self.b))
        second_term = -tf.reduce_sum(input_tensor=tf.nn.softplus(dot1(V, self.W) + self.c), axis=1)
        return first_term + second_term

    def forward_hidden(self, X):
        if False:
            i = 10
            return i + 15
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

def main():
    if False:
        return 10
    A = load_npz('Atrain.npz')
    A_test = load_npz('Atest.npz')
    mask = (A > 0) * 1.0
    mask_test = (A_test > 0) * 1.0
    (N, M) = A.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A, mask, A_test, mask_test)
if __name__ == '__main__':
    main()