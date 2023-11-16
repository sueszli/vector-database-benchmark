"""
=============================================================================
Manifold learning on handwritten digits: Locally Linear Embedding, Isomap...
=============================================================================

We illustrate various embedding techniques on the digits dataset.

"""
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
(X, y) = (digits.data, digits.target)
(n_samples, n_features) = X.shape
n_neighbors = 30
import matplotlib.pyplot as plt
(fig, axs) = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for (idx, ax) in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis('off')
_ = fig.suptitle('A selection from the 64-dimensional digits dataset', fontsize=16)
import numpy as np
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler

def plot_embedding(X, title):
    if False:
        print('Hello World!')
    (_, ax) = plt.subplots()
    X = MinMaxScaler().fit_transform(X)
    for digit in digits.target_names:
        ax.scatter(*X[y == digit].T, marker=f'${digit}$', s=60, color=plt.cm.Dark2(digit), alpha=0.425, zorder=2)
    shown_images = np.array([[1.0, 1.0]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 0.004:
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)
    ax.set_title(title)
    ax.axis('off')
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
embeddings = {'Random projection embedding': SparseRandomProjection(n_components=2, random_state=42), 'Truncated SVD embedding': TruncatedSVD(n_components=2), 'Linear Discriminant Analysis embedding': LinearDiscriminantAnalysis(n_components=2), 'Isomap embedding': Isomap(n_neighbors=n_neighbors, n_components=2), 'Standard LLE embedding': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='standard'), 'Modified LLE embedding': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='modified'), 'Hessian LLE embedding': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='hessian'), 'LTSA LLE embedding': LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='ltsa'), 'MDS embedding': MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2, normalized_stress='auto'), 'Random Trees embedding': make_pipeline(RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0), TruncatedSVD(n_components=2)), 'Spectral embedding': SpectralEmbedding(n_components=2, random_state=0, eigen_solver='arpack'), 't-SNE embedding': TSNE(n_components=2, n_iter=500, n_iter_without_progress=150, n_jobs=2, random_state=0), 'NCA embedding': NeighborhoodComponentsAnalysis(n_components=2, init='pca', random_state=0)}
from time import time
(projections, timing) = ({}, {})
for (name, transformer) in embeddings.items():
    if name.startswith('Linear Discriminant Analysis'):
        data = X.copy()
        data.flat[::X.shape[1] + 1] += 0.01
    else:
        data = X
    print(f'Computing {name}...')
    start_time = time()
    projections[name] = transformer.fit_transform(data, y)
    timing[name] = time() - start_time
for name in timing:
    title = f'{name} (time {timing[name]:.3f}s)'
    plot_embedding(projections[name], title)
plt.show()