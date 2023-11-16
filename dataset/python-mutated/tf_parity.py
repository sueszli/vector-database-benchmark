from __future__ import print_function, division
from builtins import range
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

def x2sequence(x, T, D, batch_sz):
    if False:
        while True:
            i = 10
    x = tf.transpose(x, (1, 0, 2))
    x = tf.reshape(x, (T * batch_sz, D))
    x = tf.split(x, T)
    return x

class SimpleRNN:

    def __init__(self, M):
        if False:
            while True:
                i = 10
        self.M = M

    def fit(self, X, Y, batch_sz=20, learning_rate=0.1, mu=0.9, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        if False:
            return 10
        (N, T, D) = X.shape
        K = len(set(Y.flatten()))
        M = self.M
        self.f = activation
        Wo = init_weight(M, K).astype(np.float32)
        bo = np.zeros(K, dtype=np.float32)
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)
        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')
        sequenceX = x2sequence(tfX, T, D, batch_sz)
        rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)
        (outputs, states) = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T * batch_sz, M))
        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, 1)
        targets = tf.reshape(tfY, (T * batch_sz,))
        cost_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)
        costs = []
        n_batches = N // batch_sz
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                (X, Y) = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j * batch_sz:(j + 1) * batch_sz]
                    Ybatch = Y[j * batch_sz:(j + 1) * batch_sz]
                    (_, c, p) = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c
                    for b in range(batch_sz):
                        idx = (b + 1) * T - 1
                        n_correct += p[idx] == Ybatch[b][-1]
                if i % 10 == 0:
                    print('i:', i, 'cost:', cost, 'classification rate:', float(n_correct) / N)
                if n_correct == N:
                    print('i:', i, 'cost:', cost, 'classification rate:', float(n_correct) / N)
                    break
                costs.append(cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

def parity(B=12, learning_rate=1.0, epochs=1000):
    if False:
        return 10
    (X, Y) = all_parity_pairs_with_sequence_labels(B)
    rnn = SimpleRNN(4)
    rnn.fit(X, Y, batch_sz=len(Y), learning_rate=learning_rate, epochs=epochs, activation=tf.nn.sigmoid, show_fig=False)
if __name__ == '__main__':
    parity()