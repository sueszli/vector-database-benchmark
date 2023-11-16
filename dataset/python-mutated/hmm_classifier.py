from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import string
import numpy as np
import matplotlib.pyplot as plt
from hmmd_theano2 import HMM
from sklearn.utils import shuffle
from nltk import pos_tag, word_tokenize

class HMMClassifier:

    def __init__(self):
        if False:
            return 10
        pass

    def fit(self, X, Y, V):
        if False:
            for i in range(10):
                print('nop')
        K = len(set(Y))
        N = len(Y)
        self.models = []
        self.priors = []
        for k in range(K):
            thisX = [x for (x, y) in zip(X, Y) if y == k]
            C = len(thisX)
            self.priors.append(np.log(C) - np.log(N))
            hmm = HMM(5)
            hmm.fit(thisX, V=V, print_period=1, learning_rate=0.01, max_iter=80)
            self.models.append(hmm)

    def score(self, X, Y):
        if False:
            while True:
                i = 10
        N = len(Y)
        correct = 0
        for (x, y) in zip(X, Y):
            lls = [hmm.log_likelihood(x) + prior for (hmm, prior) in zip(self.models, self.priors)]
            p = np.argmax(lls)
            if p == y:
                correct += 1
        return float(correct) / N

def get_tags(s):
    if False:
        return 10
    tuples = pos_tag(word_tokenize(s))
    return [y for (x, y) in tuples]

def get_data():
    if False:
        return 10
    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for (fn, label) in zip(('robert_frost.txt', 'edgar_allan_poe.txt'), (0, 1)):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print(line)
                tokens = get_tags(line)
                if len(tokens) > 1:
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    print(count)
                    if count >= 50:
                        break
    print('Vocabulary:', word2idx.keys())
    return (X, Y, current_idx)

def main():
    if False:
        print('Hello World!')
    (X, Y, V) = get_data()
    print('len(X):', len(X))
    print('Vocabulary size:', V)
    (X, Y) = shuffle(X, Y)
    N = 20
    (Xtrain, Ytrain) = (X[:-N], Y[:-N])
    (Xtest, Ytest) = (X[-N:], Y[-N:])
    model = HMMClassifier()
    model.fit(Xtrain, Ytrain, V)
    print('Score:', model.score(Xtest, Ytest))
if __name__ == '__main__':
    main()