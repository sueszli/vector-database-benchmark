"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn API can be used to cluster
documents by topics using a `Bag of Words approach
<https://en.wikipedia.org/wiki/Bag-of-words_model>`_.

Two algorithms are demonstrated, namely :class:`~sklearn.cluster.KMeans` and its more
scalable variant, :class:`~sklearn.cluster.MiniBatchKMeans`. Additionally,
latent semantic analysis is used to reduce dimensionality and discover latent
patterns in the data.

This example uses two different text vectorizers: a
:class:`~sklearn.feature_extraction.text.TfidfVectorizer` and a
:class:`~sklearn.feature_extraction.text.HashingVectorizer`. See the example
notebook :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`
for more information on vectorizers and a comparison of their processing times.

For document analysis via a supervised learning approach, see the example script
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`.

"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), subset='all', categories=categories, shuffle=True, random_state=42)
labels = dataset.target
(unique_labels, category_sizes) = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]
print(f'{len(dataset.data)} documents - {true_k} categories')
from collections import defaultdict
from time import time
from sklearn import metrics
evaluations = []
evaluations_std = []

def fit_and_evaluate(km, X, name=None, n_runs=5):
    if False:
        i = 10
        return i + 15
    name = km.__class__.__name__ if name is None else name
    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores['Homogeneity'].append(metrics.homogeneity_score(labels, km.labels_))
        scores['Completeness'].append(metrics.completeness_score(labels, km.labels_))
        scores['V-measure'].append(metrics.v_measure_score(labels, km.labels_))
        scores['Adjusted Rand-Index'].append(metrics.adjusted_rand_score(labels, km.labels_))
        scores['Silhouette Coefficient'].append(metrics.silhouette_score(X, km.labels_, sample_size=2000))
    train_times = np.asarray(train_times)
    print(f'clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ')
    evaluation = {'estimator': name, 'train_time': train_times.mean()}
    evaluation_std = {'estimator': name, 'train_time': train_times.std()}
    for (score_name, score_values) in scores.items():
        (mean_score, std_score) = (np.mean(score_values), np.std(score_values))
        print(f'{score_name}: {mean_score:.3f} ± {std_score:.3f}')
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english')
t0 = time()
X_tfidf = vectorizer.fit_transform(dataset.data)
print(f'vectorization done in {time() - t0:.3f} s')
print(f'n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}')
print(f'{X_tfidf.nnz / np.prod(X_tfidf.shape):.3f}')
from sklearn.cluster import KMeans
for seed in range(5):
    kmeans = KMeans(n_clusters=true_k, max_iter=100, n_init=1, random_state=seed).fit(X_tfidf)
    (cluster_ids, cluster_sizes) = np.unique(kmeans.labels_, return_counts=True)
    print(f'Number of elements assigned to each cluster: {cluster_sizes}')
print()
print(f'True number of documents in each category according to the class labels: {category_sizes}')
kmeans = KMeans(n_clusters=true_k, max_iter=100, n_init=5)
fit_and_evaluate(kmeans, X_tfidf, name='KMeans\non tf-idf vectors')
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()
print(f'LSA done in {time() - t0:.3f} s')
print(f'Explained variance of the SVD step: {explained_variance * 100:.1f}%')
kmeans = KMeans(n_clusters=true_k, max_iter=100, n_init=1)
fit_and_evaluate(kmeans, X_lsa, name='KMeans\nwith LSA on tf-idf vectors')
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=true_k, n_init=1, init_size=1000, batch_size=1000)
fit_and_evaluate(minibatch_kmeans, X_lsa, name='MiniBatchKMeans\nwith LSA on tf-idf vectors')
original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
    print(f'Cluster {i}: ', end='')
    for ind in order_centroids[i, :10]:
        print(f'{terms[ind]} ', end='')
    print()
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
lsa_vectorizer = make_pipeline(HashingVectorizer(stop_words='english', n_features=50000), TfidfTransformer(), TruncatedSVD(n_components=100, random_state=0), Normalizer(copy=False))
t0 = time()
X_hashed_lsa = lsa_vectorizer.fit_transform(dataset.data)
print(f'vectorization done in {time() - t0:.3f} s')
fit_and_evaluate(kmeans, X_hashed_lsa, name='KMeans\nwith LSA on hashed vectors')
fit_and_evaluate(minibatch_kmeans, X_hashed_lsa, name='MiniBatchKMeans\nwith LSA on hashed vectors')
import matplotlib.pyplot as plt
import pandas as pd
(fig, (ax0, ax1)) = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)
df = pd.DataFrame(evaluations[::-1]).set_index('estimator')
df_std = pd.DataFrame(evaluations_std[::-1]).set_index('estimator')
df.drop(['train_time'], axis='columns').plot.barh(ax=ax0, xerr=df_std)
ax0.set_xlabel('Clustering scores')
ax0.set_ylabel('')
df['train_time'].plot.barh(ax=ax1, xerr=df_std['train_time'])
ax1.set_xlabel('Clustering time (s)')
plt.tight_layout()