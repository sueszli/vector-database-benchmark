"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for K-means in
terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster quality
metrics to judge the goodness of fit of the cluster labels to the ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
import numpy as np
from sklearn.datasets import load_digits
(data, labels) = load_digits(return_X_y=True)
((n_samples, n_features), n_digits) = (data.shape, np.unique(labels).size)
print(f'# digits: {n_digits}; # samples: {n_samples}; # features {n_features}')
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def bench_k_means(kmeans, name, data, labels):
    if False:
        for i in range(10):
            print('nop')
    'Benchmark to evaluate the KMeans initialization methods.\n\n    Parameters\n    ----------\n    kmeans : KMeans instance\n        A :class:`~sklearn.cluster.KMeans` instance with the initialization\n        already set.\n    name : str\n        Name given to the strategy. It will be used to show the results in a\n        table.\n    data : ndarray of shape (n_samples, n_features)\n        The data to cluster.\n    labels : ndarray of shape (n_samples,)\n        The labels used to compute the clustering metrics which requires some\n        supervision.\n    '
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    clustering_metrics = [metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score, metrics.adjusted_rand_score, metrics.adjusted_mutual_info_score]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [metrics.silhouette_score(data, estimator[-1].labels_, metric='euclidean', sample_size=300)]
    formatter_result = '{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'
    print(formatter_result.format(*results))
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name='k-means++', data=data, labels=labels)
kmeans = KMeans(init='random', n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name='random', data=data, labels=labels)
pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name='PCA-based', data=data, labels=labels)
print(82 * '_')
import matplotlib.pyplot as plt
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)
h = 0.02
(x_min, x_max) = (reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1)
(y_min, y_max) = (reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1)
(xx, yy) = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\nCentroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()