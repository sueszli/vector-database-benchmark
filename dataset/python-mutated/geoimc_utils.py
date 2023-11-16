import numpy as np
from sklearn.decomposition import PCA

def length_normalize(matrix):
    if False:
        while True:
            i = 10
    'Length normalize the matrix\n\n    Args:\n        matrix (np.ndarray): Input matrix that needs to be normalized\n\n    Returns:\n        Normalized matrix\n    '
    norms = np.sqrt(np.sum(matrix ** 2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]

def mean_center(matrix):
    if False:
        i = 10
        return i + 15
    'Performs mean centering across axis 0\n\n    Args:\n        matrix (np.ndarray): Input matrix that needs to be mean centered\n    '
    avg = np.mean(matrix, axis=0)
    matrix -= avg

def reduce_dims(matrix, target_dim):
    if False:
        i = 10
        return i + 15
    'Reduce dimensionality of the data using PCA.\n\n    Args:\n        matrix (np.ndarray): Matrix of the form (n_sampes, n_features)\n        target_dim (uint): Dimension to which n_features should be reduced to.\n\n    '
    model = PCA(n_components=target_dim)
    model.fit(matrix)
    return model.transform(matrix)