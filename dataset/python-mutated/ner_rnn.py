from __future__ import print_function, division
from builtins import range
from ner_baseline import get_data
from pos_rnn import RNN

def main():
    if False:
        i = 10
        return i + 15
    (Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx) = get_data(split_sequences=True)
    V = len(word2idx)
    K = len(tag2idx)
    rnn = RNN(10, [10], V, K)
    rnn.fit(Xtrain, Ytrain, epochs=70)
    print('train score:', rnn.score(Xtrain, Ytrain))
    print('test score:', rnn.score(Xtest, Ytest))
    print('train f1 score:', rnn.f1_score(Xtrain, Ytrain))
    print('test f1 score:', rnn.f1_score(Xtest, Ytest))
if __name__ == '__main__':
    main()