from __future__ import print_function, division
from builtins import range
import os
import json
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
from util import find_analogies
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

def momentum_updates(cost, params, lr=0.0001, mu=0.9):
    if False:
        print('Hello World!')
    grads = T.grad(cost, params)
    velocities = [theano.shared(np.zeros_like(p.get_value()).astype(np.float32)) for p in params]
    updates = []
    for (p, v, g) in zip(params, velocities, grads):
        newv = mu * v - lr * g
        newp = p + newv
        updates.append((p, newp))
        updates.append((v, newv))
    return updates

class Glove:

    def __init__(self, D, V, context_sz):
        if False:
            for i in range(10):
                print('nop')
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=0.0001, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False, use_theano=False, use_tensorflow=False):
        if False:
            while True:
                i = 10
        t0 = datetime.now()
        V = self.V
        D = self.D
        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print('number of sentences to process:', N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print('processed', it, '/', N)
                n = len(sentence)
                for i in range(n):
                    wi = sentence[i]
                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)
                    if i - self.context_sz < 0:
                        points = 1.0 / (i + 1)
                        X[wi, 0] += points
                        X[0, wi] += points
                    if i + self.context_sz > n:
                        points = 1.0 / (n - i)
                        X[wi, 1] += points
                        X[1, wi] += points
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j)
                        X[wi, wj] += points
                        X[wj, wi] += points
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i)
                        X[wi, wj] += points
                        X[wj, wi] += points
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)
        print('max in X:', X.max())
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1
        print('max in f(X):', fX.max())
        logX = np.log(X + 1)
        fX = fX.astype(np.float32)
        logX = logX.astype(np.float32)
        print('max in log(X):', logX.max())
        print('time to build co-occurrence matrix:', datetime.now() - t0)
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()
        thW = theano.shared(W.astype(np.float32))
        thb = theano.shared(b.astype(np.float32))
        thU = theano.shared(U.astype(np.float32))
        thc = theano.shared(c.astype(np.float32))
        thLogX = T.matrix('logX')
        thfX = T.matrix('fX')
        params = [thW, thb, thU, thc]
        thDelta = thW.dot(thU.T) + T.reshape(thb, (V, 1)) + T.reshape(thc, (1, V)) + mu - thLogX
        thCost = (thfX * thDelta * thDelta).sum()
        regularized_cost = thCost + reg * ((thW * thW).sum() + (thU * thU).sum())
        updates = momentum_updates(regularized_cost, params, learning_rate)
        train_op = theano.function(inputs=[thfX, thLogX], updates=updates)
        cost_op = theano.function(inputs=[thfX, thLogX], outputs=thCost)
        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            train_op(fX, logX)
            cost = cost_op(fX, logX)
            costs.append(cost)
            print('epoch:', epoch, 'cost:', cost)
        self.W = thW.get_value()
        self.U = thU.get_value()
        plt.plot(costs)
        plt.show()

    def save(self, fn):
        if False:
            print('Hello World!')
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def main(we_file, w2i_file, use_brown=True, n_files=50):
    if False:
        while True:
            i = 10
    if use_brown:
        cc_matrix = 'cc_matrix_brown.npy'
    else:
        cc_matrix = 'cc_matrix_%s.npy' % n_files
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = []
    else:
        if use_brown:
            keep_words = set(['king', 'man', 'woman', 'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england', 'french', 'english', 'japan', 'japanese', 'chinese', 'italian', 'australia', 'australian', 'december', 'november', 'june', 'january', 'february', 'march', 'april', 'may', 'july', 'august', 'september', 'october'])
            (sentences, word2idx) = get_sentences_with_word2idx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            (sentences, word2idx) = get_wikipedia_data(n_files=n_files, n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)
    V = len(word2idx)
    model = Glove(100, V, 10)
    model.fit(sentences, cc_matrix=cc_matrix, learning_rate=0.0001, reg=0.1, epochs=200)
    model.save(we_file)
if __name__ == '__main__':
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    main(we, w2i, use_brown=False)
    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']
    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i: w for (w, i) in word2idx.items()}
    for concat in (True, False):
        print('** concat:', concat)
        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2
        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)