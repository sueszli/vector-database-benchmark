import numbers
import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as shuffle_
from sklearn.utils.deprecation import deprecated

@deprecated('Please import make_blobs directly from scikit-learn')
def make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    if False:
        while True:
            i = 10
    'Generate isotropic Gaussian blobs for clustering.\n\n    Read more in the :ref:`User Guide <sample_generators>`.\n\n    Parameters\n    ----------\n    n_samples : int, or tuple, optional (default=100)\n        The total number of points equally divided among clusters.\n\n    n_features : int, optional (default=2)\n        The number of features for each sample.\n\n    centers : int or array of shape [n_centers, n_features], optional\n        (default=3)\n        The number of centers to generate, or the fixed center locations.\n\n    cluster_std: float or sequence of floats, optional (default=1.0)\n        The standard deviation of the clusters.\n\n    center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))\n        The bounding box for each cluster center when centers are\n        generated at random.\n\n    shuffle : boolean, optional (default=True)\n        Shuffle the samples.\n\n    random_state : int, RandomState instance or None, optional (default=None)\n        If int, random_state is the seed used by the random number generator;\n        If RandomState instance, random_state is the random number generator;\n        If None, the random number generator is the RandomState instance used\n        by `np.random`.\n\n    Returns\n    -------\n    X : array of shape [n_samples, n_features]\n        The generated samples.\n\n    y : array of shape [n_samples]\n        The integer labels for cluster membership of each sample.\n\n    Examples\n    --------\n    >>> from sklearn.datasets.samples_generator import make_blobs\n    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,\n    ...                   random_state=0)\n    >>> print(X.shape)\n    (10, 2)\n    >>> y\n    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])\n\n    See also\n    --------\n    make_classification: a more intricate variant\n    '
    generator = check_random_state(random_state)
    if isinstance(centers, numbers.Integral):
        centers = generator.uniform(center_box[0], center_box[1], size=(centers, n_features))
    else:
        centers = check_array(centers)
        n_features = centers.shape[1]
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std
    X = []
    y = []
    n_centers = centers.shape[0]
    if isinstance(n_samples, numbers.Integral):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        n_samples_per_center = n_samples
    for (i, (n, std)) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + generator.normal(scale=std, size=(n, n_features)))
        y += [i] * n
    X = np.concatenate(X)
    y = np.array(y)
    if shuffle:
        (X, y) = shuffle_(X, y, random_state=generator)
    return (X, y)