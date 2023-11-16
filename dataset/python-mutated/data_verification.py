"""
Contains methods for verifying synthetic data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from mlfinlab.codependence import get_dependence_matrix
from mlfinlab.clustering.hierarchical_clustering import optimal_hierarchical_cluster

def plot_time_series_dependencies(time_series, dependence_method='gnpr_distance', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Plots the dependence matrix of a time series returns.\n\n    Used to verify a time series' underlying distributions via the GNPR distance method.\n    ``**kwargs`` are used to pass arguments to the `get_dependence_matrix` function used here.\n\n    :param time_series: (pd.DataFrame) Dataframe containing time series.\n    :param dependence_method: (str) Distance method to use by `get_dependence_matrix`\n    :return: (plt.Axes) Figure's axes.\n    "
    return axis

def _compute_eigenvalues(mats):
    if False:
        print('Hello World!')
    '\n    Computes the eigenvalues of each matrix.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param mats: (np.array) List of matrices to calculate eigenvalues from.\n        Has shape (n_sample, dim, dim)\n    :return: (np.array) Resulting eigenvalues from mats.\n    '
    pass

def _compute_pf_vec(mats):
    if False:
        while True:
            i = 10
    '\n    Computes the Perron-Frobenius vector of each matrix.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    The Perron-Frobenius property asserts that for a strictly positive square matrix, the\n    corresponding eigenvector of the largest eigenvalue has strictly positive components.\n\n    :param mats: (np.array) List of matrices to calculate Perron-Frobenius vector from.\n        Has shape (n_sample, dim, dim)\n    :return: (np.array) Resulting Perron-Frobenius vectors from mats.\n    '
    pass

def _compute_degree_counts(mats):
    if False:
        i = 10
        return i + 15
    '\n    Computes the number of degrees in MST of each matrix.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    The degree count is calculated by computing the MST of the matrix, and counting\n    how many times each nodes appears in each edge produced by the MST. This count is normalized\n    by the size of the matrix.\n\n    :param mats: (np.array) List of matrices to calculate the number of degrees in MST from.\n        Has shape (n_sample, dim, dim)\n    :return: (np.array) Resulting number of degrees in MST from mats.\n    '
    pass

def plot_pairwise_dist(emp_mats, gen_mats, n_hist=100):
    if False:
        while True:
            i = 10
    "\n    Plots the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    - Distribution of pairwise correlations is significantly shifted to the positive.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n        Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n        Has shape (n_samples_b, dim_b, dim_b)\n    :param n_hist: (int) Number of bins for histogram plots. (100 by default).\n    :return: (plt.Axes) Figure's axes.\n    "
    pass

def plot_eigenvalues(emp_mats, gen_mats, n_hist=100):
    if False:
        i = 10
        return i + 15
    "\n    Plots the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    - Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first eigenvalue (the market).\n\n    - Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other large eigenvalues (industries).\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n        Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n        Has shape (n_samples_b, dim_b, dim_b)\n    :param n_hist: (int) Number of bins for histogram plots. (100 by default).\n    :return: (plt.Axes) Figure's axes.\n    "
    pass

def plot_eigenvectors(emp_mats, gen_mats, n_hist=100):
    if False:
        while True:
            i = 10
    "\n    Plots the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    - Perron-Frobenius property (first eigenvector has positive entries).\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n       Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n       Has shape (n_samples_b, dim_b, dim_b)\n    :param n_hist: (int) Number of bins for histogram plots. (100 by default).\n    :return: (plt.Axes) Figure's axes.\n    "
    pass

def plot_hierarchical_structure(emp_mats, gen_mats):
    if False:
        return 10
    "\n    Plots the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    - Hierarchical structure of correlations.\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n       Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n       Has shape (n_samples_b, dim_b, dim_b)\n    :return: (tuple) Figures' axes.\n    "
    pass

def plot_mst_degree_count(emp_mats, gen_mats):
    if False:
        i = 10
        return i + 15
    "\n    Plots all the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    - Scale-free property of the corresponding Minimum Spanning Tree (MST).\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n       Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n       Has shape (n_samples_b, dim_b, dim_b)\n    :return: (plt.Axes) Figure's axes.\n    "
    pass

def plot_stylized_facts(emp_mats, gen_mats, n_hist=100):
    if False:
        print('Hello World!')
    '\n    Plots the following stylized facts for comparison between empirical and generated\n    correlation matrices:\n\n    1. Distribution of pairwise correlations is significantly shifted to the positive.\n\n    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first\n    eigenvalue (the market).\n\n    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other\n    large eigenvalues (industries).\n\n    4. Perron-Frobenius property (first eigenvector has positive entries).\n\n    5. Hierarchical structure of correlations.\n\n    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).\n\n    It is reproduced with modifications from the following blog post:\n    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].\n    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.\n    (Accessed: 17 Aug 2020)\n    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_\n\n    :param emp_mats: (np.array) Empirical correlation matrices.\n        Has shape (n_samples_a, dim_a, dim_a)\n    :param gen_mats: (np.array) Generated correlation matrices.\n        Has shape (n_samples_b, dim_b, dim_b)\n    :param n_hist: (int) Number of bins for histogram plots. (100 by default).\n    '
    pass

def plot_optimal_hierarchical_cluster(mat, method='ward'):
    if False:
        while True:
            i = 10
    '\n    Calculates and plots the optimal clustering of a correlation matrix.\n\n    It uses the `optimal_hierarchical_cluster` function in the clustering module to calculate\n    the optimal hierarchy cluster matrix.\n\n    :param mat: (np.array/pd.DataFrame) Correlation matrix.\n    :param method: (str) Method to calculate the hierarchy clusters. Can take the values\n        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].\n    :return: (plt.Axes) Figure\'s axes.\n    '
    pass