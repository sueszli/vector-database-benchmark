import os
import numpy as np
(this_dir, this_filename) = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, 'data', 'three_blobs.csv.gz')

def three_blobs_data():
    if False:
        return 10
    'A random dataset of 3 2D blobs for clustering.\n\n    Number of samples : 150\n    Suggested labels : {0, 1, 2}, distribution: [50, 50, 50]\n\n    Returns\n    --------\n    X, y : [n_samples, n_features], [n_cluster_labels]\n        X is the feature matrix with 159 samples as rows\n        and 2 feature columns.\n        y is a 1-dimensional array of the 3 suggested cluster labels 0, 1, 2\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/data/three_blobs_data\n\n    '
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
    (X, y) = (tmp[:, :-1], tmp[:, -1])
    y = y.astype(int)
    return (X, y)