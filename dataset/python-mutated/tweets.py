from __future__ import print_function, division
from builtins import range
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = ['the', 'about', 'an', 'and', 'are', 'at', 'be', 'can', 'for', 'from', 'if', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'this', 'to', 'you', 'your', 'with']
url_finder = re.compile('(?:\\@|https?\\://)\\S+')

def filter_tweet(s):
    if False:
        for i in range(10):
            print('nop')
    s = s.lower()
    s = url_finder.sub('', s)
    return s
df = pd.read_csv('../large_files/tweets.csv')
text = df.text.tolist()
text = [filter_tweet(s) for s in text]
tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)
X = tfidf.fit_transform(text).todense()
N = X.shape[0]
idx = np.random.choice(N, size=2000, replace=False)
x = X[idx]
labels = df.handle[idx].tolist()
pTrump = sum((1.0 if e == 'realDonaldTrump' else 0.0 for e in labels)) / len(labels)
print('proportion @realDonaldTrump: %.3f' % pTrump)
print('proportion @HillaryClinton: %.3f' % (1 - pTrump))
dist_array = pdist(x)
Z = linkage(dist_array, 'ward')
plt.title('Ward')
dendrogram(Z, labels=labels)
plt.show()
Y = np.array([1 if e == 'realDonaldTrump' else 2 for e in labels])
C = fcluster(Z, 9, criterion='distance')
categories = set(C)
print('values in C:', categories)

def purity(true_labels, cluster_assignments, categories):
    if False:
        return 10
    N = len(true_labels)
    total = 0.0
    for k in categories:
        max_intersection = 0
        for j in categories:
            intersection = ((cluster_assignments == k) & (true_labels == j)).sum()
            if intersection > max_intersection:
                max_intersection = intersection
        total += max_intersection
    return total / N
print('purity:', purity(Y, C, categories))
if (C == 1).sum() < (C == 2).sum():
    d = 1
    h = 2
else:
    d = 2
    h = 1
actually_donald = ((C == d) & (Y == 1)).sum()
donald_cluster_size = (C == d).sum()
print('purity of @realDonaldTrump cluster:', float(actually_donald) / donald_cluster_size)
actually_hillary = ((C == h) & (Y == 2)).sum()
hillary_cluster_size = (C == h).sum()
print('purity of @HillaryClinton cluster:', float(actually_hillary) / hillary_cluster_size)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, df.handle)
print('classifier score:', rf.score(X, df.handle))
w2i = tfidf.vocabulary_
d_avg = np.array(x[C == d].mean(axis=0)).flatten()
d_sorted = sorted(w2i.keys(), key=lambda w: -d_avg[w2i[w]])
print("\nTop 10 'Donald cluster' words:")
print('\n'.join(d_sorted[:10]))
h_avg = np.array(x[C == h].mean(axis=0)).flatten()
h_sorted = sorted(w2i.keys(), key=lambda w: -h_avg[w2i[w]])
print("\nTop 10 'Hillary cluster' words:")
print('\n'.join(h_sorted[:10]))