"""
=====================================
Approximate nearest neighbors in TSNE
=====================================

This example presents how to chain KNeighborsTransformer and TSNE in a pipeline.
It also shows how to wrap the packages `nmslib` and `pynndescent` to replace
KNeighborsTransformer and perform approximate nearest neighbors. These packages
can be installed with `pip install nmslib pynndescent`.

Note: In KNeighborsTransformer we use the definition which includes each
training point as its own neighbor in the count of `n_neighbors`, and for
compatibility reasons, one extra neighbor is computed when `mode == 'distance'`.
Please note that we do the same in the proposed `nmslib` wrapper.
"""
import sys
try:
    import nmslib
except ImportError:
    print("The package 'nmslib' is required to run this example.")
    sys.exit()
try:
    from pynndescent import PyNNDescentTransformer
except ImportError:
    print("The package 'pynndescent' is required to run this example.")
    sys.exit()
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric='euclidean', method='sw-graph', n_jobs=-1):
        if False:
            i = 10
            return i + 15
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        if False:
            print('Hello World!')
        self.n_samples_fit_ = X.shape[0]
        space = {'euclidean': 'l2', 'cosine': 'cosinesimil', 'l1': 'l1', 'l2': 'l2'}[self.metric]
        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X.copy())
        self.nmslib_.createIndex()
        return self

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        n_samples_transform = X.shape[0]
        n_neighbors = self.n_neighbors + 1
        if self.n_jobs < 0:
            num_threads = joblib.cpu_count() + self.n_jobs + 1
        else:
            num_threads = self.n_jobs
        results = self.nmslib_.knnQueryBatch(X.copy(), k=n_neighbors, num_threads=num_threads)
        (indices, distances) = zip(*results)
        (indices, distances) = (np.vstack(indices), np.vstack(distances))
        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(), indptr), shape=(n_samples_transform, self.n_samples_fit_))
        return kneighbors_graph

def load_mnist(n_samples):
    if False:
        return 10
    'Load MNIST, shuffle the data, and return only n_samples.'
    mnist = fetch_openml('mnist_784', as_frame=False, parser='pandas')
    (X, y) = shuffle(mnist.data, mnist.target, random_state=2)
    return (X[:n_samples] / 255, y[:n_samples])
import time
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsTransformer
from sklearn.pipeline import make_pipeline
datasets = [('MNIST_10000', load_mnist(n_samples=10000)), ('MNIST_20000', load_mnist(n_samples=20000))]
n_iter = 500
perplexity = 30
metric = 'euclidean'
n_neighbors = int(3.0 * perplexity + 1) + 1
tsne_params = dict(init='random', perplexity=perplexity, method='barnes_hut', random_state=42, n_iter=n_iter, learning_rate='auto')
transformers = [('KNeighborsTransformer', KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance', metric=metric)), ('NMSlibTransformer', NMSlibTransformer(n_neighbors=n_neighbors, metric=metric)), ('PyNNDescentTransformer', PyNNDescentTransformer(n_neighbors=n_neighbors, metric=metric, parallel_batch_queries=True))]
for (dataset_name, (X, y)) in datasets:
    msg = f'Benchmarking on {dataset_name}:'
    print(f'\n{msg}\n' + str('-' * len(msg)))
    for (transformer_name, transformer) in transformers:
        longest = np.max([len(name) for (name, model) in transformers])
        start = time.time()
        transformer.fit(X)
        fit_duration = time.time() - start
        print(f'{transformer_name:<{longest}} {fit_duration:.3f} sec (fit)')
        start = time.time()
        Xt = transformer.transform(X)
        transform_duration = time.time() - start
        print(f'{transformer_name:<{longest}} {transform_duration:.3f} sec (transform)')
        if transformer_name == 'PyNNDescentTransformer':
            start = time.time()
            Xt = transformer.transform(X)
            transform_duration = time.time() - start
            print(f'{transformer_name:<{longest}} {transform_duration:.3f} sec (transform)')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
transformers = [('TSNE with internal NearestNeighbors', TSNE(metric=metric, **tsne_params)), ('TSNE with KNeighborsTransformer', make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='distance', metric=metric), TSNE(metric='precomputed', **tsne_params))), ('TSNE with NMSlibTransformer', make_pipeline(NMSlibTransformer(n_neighbors=n_neighbors, metric=metric), TSNE(metric='precomputed', **tsne_params)))]
nrows = len(datasets)
ncols = np.sum([1 for (name, model) in transformers if 'TSNE' in name])
(fig, axes) = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(5 * ncols, 4 * nrows))
axes = axes.ravel()
i_ax = 0
for (dataset_name, (X, y)) in datasets:
    msg = f'Benchmarking on {dataset_name}:'
    print(f'\n{msg}\n' + str('-' * len(msg)))
    for (transformer_name, transformer) in transformers:
        longest = np.max([len(name) for (name, model) in transformers])
        start = time.time()
        Xt = transformer.fit_transform(X)
        transform_duration = time.time() - start
        print(f'{transformer_name:<{longest}} {transform_duration:.3f} sec (fit_transform)')
        axes[i_ax].set_title(transformer_name + '\non ' + dataset_name)
        axes[i_ax].scatter(Xt[:, 0], Xt[:, 1], c=y.astype(np.int32), alpha=0.2, cmap=plt.cm.viridis)
        axes[i_ax].xaxis.set_major_formatter(NullFormatter())
        axes[i_ax].yaxis.set_major_formatter(NullFormatter())
        axes[i_ax].axis('tight')
        i_ax += 1
fig.tight_layout()
plt.show()