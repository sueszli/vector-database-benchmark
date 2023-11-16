from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import init_weight, get_robert_frost, get_wikipedia_data

class SimpleRNN:

    def __init__(self, D, M, V, f, session):
        if False:
            while True:
                i = 10
        self.D = D
        self.M = M
        self.V = V
        self.f = f
        self.session = session

    def set_session(self, session):
        if False:
            while True:
                i = 10
        self.session = session

    def build(self, We, Wx, Wh, bh, h0, Wo, bo):
        if False:
            while True:
                i = 10
        self.We = tf.Variable(We)
        self.Wx = tf.Variable(Wx)
        self.Wh = tf.Variable(Wh)
        self.bh = tf.Variable(bh)
        self.h0 = tf.Variable(h0)
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        V = self.V
        D = self.D
        M = self.M
        self.tfX = tf.placeholder(tf.int32, shape=(None,), name='X')
        self.tfY = tf.placeholder(tf.int32, shape=(None,), name='Y')
        XW = tf.nn.embedding_lookup(We, self.tfX)
        XW_Wx = tf.matmul(XW, self.Wx)

        def recurrence(h_t1, XW_Wx_t):
            if False:
                while True:
                    i = 10
            h_t1 = tf.reshape(h_t1, (1, M))
            h_t = self.f(XW_Wx_t + tf.matmul(h_t1, self.Wh) + self.bh)
            h_t = tf.reshape(h_t, (M,))
            return h_t
        h = tf.scan(fn=recurrence, elems=XW_Wx, initializer=self.h0)
        logits = tf.matmul(h, self.Wo) + self.bo
        prediction = tf.argmax(logits, 1)
        self.output_probs = tf.nn.softmax(logits)
        nce_weights = tf.transpose(self.Wo, [1, 0])
        nce_biases = self.bo
        h = tf.reshape(h, (-1, M))
        labels = tf.reshape(self.tfY, (-1, 1))
        self.cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, labels=labels, inputs=h, num_sampled=50, num_classes=V))
        self.predict_op = prediction
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.session.run(init)

    def fit(self, X, epochs=500, show_fig=False):
        if False:
            while True:
                i = 10
        N = len(X)
        D = self.D
        M = self.M
        V = self.V
        We = init_weight(V, D).astype(np.float32)
        Wx = init_weight(D, M).astype(np.float32)
        Wh = init_weight(M, M).astype(np.float32)
        bh = np.zeros(M).astype(np.float32)
        h0 = np.zeros(M).astype(np.float32)
        Wo = init_weight(M, V).astype(np.float32)
        bo = np.zeros(V).astype(np.float32)
        self.build(We, Wx, Wh, bh, h0, Wo, bo)
        costs = []
        n_total = sum((len(sentence) + 1 for sentence in X))
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                input_sequence = [0] + X[j]
                output_sequence = X[j] + [1]
                (_, c, p) = self.session.run((self.train_op, self.cost, self.predict_op), feed_dict={self.tfX: input_sequence, self.tfY: output_sequence})
                cost += c
                for (pj, xj) in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print('i:', i, 'cost:', cost, 'correct rate:', float(n_correct) / n_total)
            costs.append(cost)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def predict(self, prev_words):
        if False:
            for i in range(10):
                print('nop')
        return self.session.run(self.output_probs, feed_dict={self.tfX: prev_words})

    def save(self, filename):
        if False:
            print('Hello World!')
        actual_params = self.session.run(self.params)
        np.savez(filename, *[p for p in actual_params])

    @staticmethod
    def load(filename, activation, session):
        if False:
            return 10
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
        (V, D) = We.shape
        (_, M) = Wx.shape
        rnn = SimpleRNN(D, M, V, activation, session)
        rnn.build(We, Wx, Wh, bh, h0, Wo, bo)
        return rnn

    def generate(self, pi, word2idx):
        if False:
            i = 10
            return i + 15
        idx2word = {v: k for (k, v) in iteritems(word2idx)}
        V = len(pi)
        n_lines = 0
        X = [np.random.choice(V, p=pi)]
        print(idx2word[X[0]], end=' ')
        while n_lines < 4:
            probs = self.predict(X)[-1]
            word_idx = np.random.choice(V, p=probs)
            X.append(word_idx)
            if word_idx > 1:
                word = idx2word[word_idx]
                print(word, end=' ')
            elif word_idx == 1:
                n_lines += 1
                print('')
                if n_lines < 4:
                    X = [np.random.choice(V, p=pi)]
                    print(idx2word[X[0]], end=' ')

def train_poetry(session, dims, savefile):
    if False:
        print('Hello World!')
    (sentences, word2idx) = get_robert_frost()
    rnn = SimpleRNN(dims, dims, len(word2idx), tf.nn.relu, session)
    rnn.fit(sentences, epochs=17, show_fig=True)
    rnn.save(savefile)

def generate_poetry(session, savefile):
    if False:
        print('Hello World!')
    (sentences, word2idx) = get_robert_frost()
    rnn = SimpleRNN.load(savefile, tf.nn.relu, session)
    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi /= pi.sum()
    rnn.generate(pi, word2idx)
if __name__ == '__main__':
    dims = 50
    savefile = 'RNN_D50_M50_tf.npz'
    session = tf.InteractiveSession()
    train_poetry(session, dims, savefile)
    generate_poetry(session, savefile)