"""
Implementation of hierarchical clustering algorithms.
"""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

def optimal_hierarchical_cluster(mat: np.array, method: str='ward') -> np.array:
    if False:
        while True:
            i = 10
    '\n    Calculates the optimal clustering of a matrix.\n\n    It calculates the hierarchy clusters from the distance of the matrix. Then it calculates\n    the optimal leaf ordering of the hierarchy clusters, and returns the optimally clustered matrix.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    This method relies and acts as a wrapper for the `scipy.cluster.hierarchy` module.\n    `<https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_\n\n    :param mat: (np.array/pd.DataFrame) Correlation matrix.\n    :param method: (str) Method to calculate the hierarchy clusters. Can take the values\n        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].\n    :return: (np.array) Optimal hierarchy cluster matrix.\n    '
    pass