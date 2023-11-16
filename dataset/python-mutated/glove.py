from __future__ import print_function, division
from builtins import range
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
from util import find_analogies
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

class Glove:

    def __init__(self, D, V, context_sz):
        if False:
            return 10
        self.D = D
        self.V = V
        self.context_sz = context_sz

    def fit(self, sentences, cc_matrix=None, learning_rate=0.0001, reg=0.1, xmax=100, alpha=0.75, epochs=10, gd=False):
        if False:
            for i in range(10):
                print('nop')
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
        print('max in log(X):', logX.max())
        print('time to build co-occurrence matrix:', datetime.now() - t0)
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()
        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            cost = (fX * delta * delta).sum()
            costs.append(cost)
            print('epoch:', epoch, 'cost:', cost)
            if gd:
                for i in range(V):
                    W[i] -= learning_rate * (fX[i, :] * delta[i, :]).dot(U)
                W -= learning_rate * reg * W
                for i in range(V):
                    b[i] -= learning_rate * fX[i, :].dot(delta[i, :])
                for j in range(V):
                    U[j] -= learning_rate * (fX[:, j] * delta[:, j]).dot(W)
                U -= learning_rate * reg * U
                for j in range(V):
                    c[j] -= learning_rate * fX[:, j].dot(delta[:, j])
            else:
                for i in range(V):
                    matrix = reg * np.eye(D) + (fX[i, :] * U.T).dot(U)
                    vector = (fX[i, :] * (logX[i, :] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector)
                for i in range(V):
                    denominator = fX[i, :].sum() + reg
                    numerator = fX[i, :].dot(logX[i, :] - W[i].dot(U.T) - c - mu)
                    b[i] = numerator / denominator
                for j in range(V):
                    matrix = reg * np.eye(D) + (fX[:, j] * W.T).dot(W)
                    vector = (fX[:, j] * (logX[:, j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)
                for j in range(V):
                    denominator = fX[:, j].sum() + reg
                    numerator = fX[:, j].dot(logX[:, j] - W.dot(U[j]) - b - mu)
                    c[j] = numerator / denominator
        self.W = W
        self.U = U
        plt.plot(costs)
        plt.show()

    def save(self, fn):
        if False:
            while True:
                i = 10
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)

def main(we_file, w2i_file, use_brown=True, n_files=100):
    if False:
        return 10
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
    model.fit(sentences, cc_matrix=cc_matrix, epochs=20)
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