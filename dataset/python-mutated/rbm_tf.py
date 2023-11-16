from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getKaggleMNIST
from autoencoder_tf import DNN

class RBM(object):

    def __init__(self, D, M, an_id):
        if False:
            while True:
                i = 10
        self.D = D
        self.M = M
        self.id = an_id
        self.build(D, M)

    def set_session(self, session):
        if False:
            return 10
        self.session = session

    def build(self, D, M):
        if False:
            return 10
        self.W = tf.Variable(tf.random.normal(shape=(D, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros(D).astype(np.float32))
        self.X_in = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v
        r = tf.random.uniform(shape=tf.shape(input=p_h_given_v))
        H = tf.cast(r < p_h_given_v, dtype=tf.float32)
        p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(a=self.W)) + self.b)
        r = tf.random.uniform(shape=tf.shape(input=p_v_given_h))
        X_sample = tf.cast(r < p_v_given_h, dtype=tf.float32)
        objective = tf.reduce_mean(input_tensor=self.free_energy(self.X_in)) - tf.reduce_mean(input_tensor=self.free_energy(X_sample))
        self.train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(objective)
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X_in, logits=logits))

    def fit(self, X, epochs=1, batch_sz=100, show_fig=False):
        if False:
            print('Hello World!')
        (N, D) = X.shape
        n_batches = N // batch_sz
        costs = []
        print('training rbm: %s' % self.id)
        for i in range(epochs):
            print('epoch:', i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:j * batch_sz + batch_sz]
                (_, c) = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                if j % 10 == 0:
                    print('j / n_batches:', j, '/', n_batches, 'cost:', c)
                costs.append(c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def free_energy(self, V):
        if False:
            for i in range(10):
                print('nop')
        b = tf.reshape(self.b, (self.D, 1))
        first_term = -tf.matmul(V, b)
        first_term = tf.reshape(first_term, (-1,))
        second_term = -tf.reduce_sum(input_tensor=tf.nn.softplus(tf.matmul(V, self.W) + self.c), axis=1)
        return first_term + second_term

    def forward_hidden(self, X):
        if False:
            i = 10
            return i + 15
        return tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)

    def forward_logits(self, X):
        if False:
            print('Hello World!')
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(a=self.W)) + self.b

    def forward_output(self, X):
        if False:
            for i in range(10):
                print('nop')
        return tf.nn.sigmoid(self.forward_logits(X))

    def transform(self, X):
        if False:
            print('Hello World!')
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})

def main():
    if False:
        for i in range(10):
            print('nop')
    (Xtrain, Ytrain, Xtest, Ytest) = getKaggleMNIST()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    (_, D) = Xtrain.shape
    K = len(set(Ytrain))
    dnn = DNN(D, [1000, 750, 500], K, UnsupervisedModel=RBM)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        session.run(init_op)
        dnn.set_session(session)
        dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=True, epochs=10)
if __name__ == '__main__':
    main()