from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from util import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer

def main():
    if False:
        print('Hello World!')
    analogies_to_try = (('king', 'man', 'woman'), ('france', 'paris', 'london'), ('france', 'paris', 'rome'), ('paris', 'france', 'italy'))
    (sentences, word2idx) = get_wikipedia_data(n_files=3, n_vocab=2000, by_paragraph=True)
    notfound = False
    for word_list in analogies_to_try:
        for w in word_list:
            if w not in word2idx:
                print('%s not found in vocab, remove it from                     analogies to try or increase vocab size' % w)
                notfound = True
    if notfound:
        exit()
    V = len(word2idx)
    N = len(sentences)
    A = np.zeros((V, N))
    print('V:', V, 'N:', N)
    j = 0
    for sentence in sentences:
        for i in sentence:
            A[i, j] += 1
        j += 1
    print('finished getting raw counts')
    transformer = TfidfTransformer()
    A = transformer.fit_transform(A.T).T
    A = A.toarray()
    idx2word = {v: k for (k, v) in iteritems(word2idx)}
    tsne = TSNE()
    Z = tsne.fit_transform(A)
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(V):
        try:
            plt.annotate(s=idx2word[i].encode('utf8').decode('utf8'), xy=(Z[i, 0], Z[i, 1]))
        except:
            print('bad string:', idx2word[i])
    plt.draw()
    tsne = TSNE(n_components=3)
    We = tsne.fit_transform(A)
    for word_list in analogies_to_try:
        (w1, w2, w3) = word_list
        find_analogies(w1, w2, w3, We, word2idx, idx2word)
    plt.show()
if __name__ == '__main__':
    main()