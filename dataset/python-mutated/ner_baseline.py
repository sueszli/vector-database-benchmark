from __future__ import print_function, division
from builtins import range
import numpy as np
from sklearn.utils import shuffle
from pos_baseline import LogisticRegression

def get_data(split_sequences=False):
    if False:
        for i in range(10):
            print('nop')
    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open('ner.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            (word, tag) = r
            word = word.lower()
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])
            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY
    print('number of samples:', len(Xtrain))
    (Xtrain, Ytrain) = shuffle(Xtrain, Ytrain)
    Ntest = int(0.3 * len(Xtrain))
    Xtest = Xtrain[:Ntest]
    Ytest = Ytrain[:Ntest]
    Xtrain = Xtrain[Ntest:]
    Ytrain = Ytrain[Ntest:]
    print('number of classes:', len(tag2idx))
    return (Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx)

def main():
    if False:
        print('Hello World!')
    (Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx) = get_data()
    V = len(word2idx)
    print('vocabulary size:', V)
    K = len(tag2idx)
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V, K=K, epochs=5)
    print('training complete')
    print('train score:', model.score(Xtrain, Ytrain))
    print('train f1 score:', model.f1_score(Xtrain, Ytrain))
    print('test score:', model.score(Xtest, Ytest))
    print('test f1 score:', model.f1_score(Xtest, Ytest))
if __name__ == '__main__':
    main()