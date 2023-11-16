from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import networkx as nx
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding as LLE
from sklearn.feature_extraction.text import TfidfTransformer
wordnet_lemmatizer = WordNetLemmatizer()
titles = [line.rstrip() for line in open('../nlp_class/all_book_titles.txt')]
stopwords = set((w.rstrip() for w in open('../nlp_class/stopwords.txt')))
stopwords = stopwords.union({'introduction', 'edition', 'series', 'application', 'approach', 'card', 'access', 'package', 'plus', 'etext', 'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed', 'third', 'second', 'fourth'})

def my_tokenizer(s):
    if False:
        for i in range(10):
            print('nop')
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if not any((c.isdigit() for c in t))]
    return tokens
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
print('num titles:', len(titles))
print('first title:', titles[0])
for title in titles:
    try:
        title = title.encode('ascii', 'ignore')
        title = title.decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        print(e)

def tokens_to_vector(tokens):
    if False:
        return 10
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    return x
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in all_tokens:
    X[:, i] = tokens_to_vector(tokens)
    i += 1

def d(u, v):
    if False:
        for i in range(10):
            print('nop')
    diff = u - v
    return diff.dot(diff)

def cost(X, R, M):
    if False:
        print('Hello World!')
    cost = 0
    for k in range(len(M)):
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:, k] * sq_distances).sum()
    return cost

def plot_k_means(X, K, index_word_map, max_iter=20, beta=1.0, show_plots=True):
    if False:
        return 10
    (N, D) = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    exponents = np.empty((N, K))
    for k in range(K):
        M[k] = X[np.random.choice(N)]
    costs = np.zeros(max_iter)
    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                exponents[n, k] = np.exp(-beta * d(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)
        for k in range(K):
            M[k] = R[:, k].dot(X) / R[:, k].sum()
        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i - 1]) < 0.0001:
                break
    if show_plots:
        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.figure(figsize=(80.0, 80.0))
        plt.scatter(X[:, 0], X[:, 1], s=300, alpha=0.9, c=colors)
        annotate1(X, index_word_map)
        plt.savefig('test.png')
    hard_responsibilities = np.argmax(R, axis=1)
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
        word = index_word_map[i]
        cluster = hard_responsibilities[i]
        if cluster not in cluster2word:
            cluster2word[cluster] = []
        cluster2word[cluster].append(word)
    for (cluster, wordlist) in cluster2word.items():
        print('cluster', cluster, '->', wordlist)
    return (M, R)

def annotate1(X, index_word_map, eps=0.1):
    if False:
        for i in range(10):
            print('nop')
    (N, D) = X.shape
    placed = np.empty((N, D))
    for i in range(N):
        (x, y) = X[i]
        close = []
        (x, y) = X[i]
        for retry in range(3):
            for j in range(i):
                diff = np.array([x, y]) - placed[j]
                if diff.dot(diff) < eps:
                    close.append(placed[j])
            if close:
                x += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
                y += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
                close = []
            else:
                break
        placed[i] = (x, y)
        plt.annotate(s=index_word_map[i], xy=(X[i, 0], X[i, 1]), xytext=(x, y), arrowprops={'arrowstyle': '->', 'color': 'black'})
print('vocab size:', current_index)
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()
reducer = TSNE()
Z = reducer.fit_transform(X)
plot_k_means(Z[:, :2], current_index // 10, index_word_map, show_plots=True)