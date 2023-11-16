from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath('..'))
from hmm_class.hmmd_scaled import HMM
from pos_baseline import get_data
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score

def accuracy(T, Y):
    if False:
        return 10
    n_correct = 0
    n_total = 0
    for (t, y) in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total

def total_f1_score(T, Y):
    if False:
        while True:
            i = 10
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()

def main(smoothing=0.1):
    if False:
        return 10
    (Xtrain, Ytrain, Xtest, Ytest, word2idx) = get_data(split_sequences=True)
    V = len(word2idx) + 1
    M = max((max(y) for y in Ytrain)) + 1
    A = np.ones((M, M)) * smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        pi[y[0]] += 1
        for i in range(len(y) - 1):
            A[y[i], y[i + 1]] += 1
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()
    B = np.ones((M, V)) * smoothing
    for (x, y) in zip(Xtrain, Ytrain):
        for (xi, yi) in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis=1, keepdims=True)
    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B
    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)
    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)
    print('train accuracy:', accuracy(Ytrain, Ptrain))
    print('test accuracy:', accuracy(Ytest, Ptest))
    print('train f1:', total_f1_score(Ytrain, Ptrain))
    print('test f1:', total_f1_score(Ytest, Ptest))
if __name__ == '__main__':
    main()