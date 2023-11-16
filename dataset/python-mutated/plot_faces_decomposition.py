"""
============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces_dataset` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`).


- Authors: Vlad Niculae, Alexandre Gramfort
- License: BSD 3 clause
"""
import logging
import matplotlib.pyplot as plt
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
rng = RandomState(0)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
(faces, _) = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
(n_samples, n_features) = faces.shape
faces_centered = faces - faces.mean(axis=0)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
print('Dataset consists of %d faces' % n_samples)
(n_row, n_col) = (2, 3)
n_components = n_row * n_col
image_shape = (64, 64)

def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    if False:
        i = 10
        return i + 15
    (fig, axs) = plt.subplots(nrows=n_row, ncols=n_col, figsize=(2.0 * n_col, 2.3 * n_row), facecolor='white', constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor('black')
    fig.suptitle(title, size=16)
    for (ax, vec) in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(vec.reshape(image_shape), cmap=cmap, interpolation='nearest', vmin=-vmax, vmax=vmax)
        ax.axis('off')
    fig.colorbar(im, ax=axs, orientation='horizontal', shrink=0.99, aspect=40, pad=0.01)
    plt.show()
plot_gallery('Faces from dataset', faces_centered[:n_components])
pca_estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca_estimator.fit(faces_centered)
plot_gallery('Eigenfaces - PCA using randomized SVD', pca_estimator.components_[:n_components])
nmf_estimator = decomposition.NMF(n_components=n_components, tol=0.005)
nmf_estimator.fit(faces)
plot_gallery('Non-negative components - NMF', nmf_estimator.components_[:n_components])
ica_estimator = decomposition.FastICA(n_components=n_components, max_iter=400, whiten='arbitrary-variance', tol=0.00015)
ica_estimator.fit(faces_centered)
plot_gallery('Independent components - FastICA', ica_estimator.components_[:n_components])
batch_pca_estimator = decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.1, max_iter=100, batch_size=3, random_state=rng)
batch_pca_estimator.fit(faces_centered)
plot_gallery('Sparse components - MiniBatchSparsePCA', batch_pca_estimator.components_[:n_components])
batch_dict_estimator = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1, max_iter=50, batch_size=3, random_state=rng)
batch_dict_estimator.fit(faces_centered)
plot_gallery('Dictionary learning', batch_dict_estimator.components_[:n_components])
kmeans_estimator = cluster.MiniBatchKMeans(n_clusters=n_components, tol=0.001, batch_size=20, max_iter=50, random_state=rng, n_init='auto')
kmeans_estimator.fit(faces_centered)
plot_gallery('Cluster centers - MiniBatchKMeans', kmeans_estimator.cluster_centers_[:n_components])
fa_estimator = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
fa_estimator.fit(faces_centered)
plot_gallery('Factor Analysis (FA)', fa_estimator.components_[:n_components])
plt.figure(figsize=(3.2, 3.6), facecolor='white', tight_layout=True)
vec = fa_estimator.noise_variance_
vmax = max(vec.max(), -vec.min())
plt.imshow(vec.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax)
plt.axis('off')
plt.title('Pixelwise variance from \n Factor Analysis (FA)', size=16, wrap=True)
plt.colorbar(orientation='horizontal', shrink=0.8, pad=0.03)
plt.show()
plot_gallery('Faces from dataset', faces_centered[:n_components], cmap=plt.cm.RdBu)
dict_pos_dict_estimator = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1, max_iter=50, batch_size=3, random_state=rng, positive_dict=True)
dict_pos_dict_estimator.fit(faces_centered)
plot_gallery('Dictionary learning - positive dictionary', dict_pos_dict_estimator.components_[:n_components], cmap=plt.cm.RdBu)
dict_pos_code_estimator = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1, max_iter=50, batch_size=3, fit_algorithm='cd', random_state=rng, positive_code=True)
dict_pos_code_estimator.fit(faces_centered)
plot_gallery('Dictionary learning - positive code', dict_pos_code_estimator.components_[:n_components], cmap=plt.cm.RdBu)
dict_pos_estimator = decomposition.MiniBatchDictionaryLearning(n_components=n_components, alpha=0.1, max_iter=50, batch_size=3, fit_algorithm='cd', random_state=rng, positive_dict=True, positive_code=True)
dict_pos_estimator.fit(faces_centered)
plot_gallery('Dictionary learning - positive dictionary & code', dict_pos_estimator.components_[:n_components], cmap=plt.cm.RdBu)