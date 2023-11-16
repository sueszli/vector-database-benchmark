"""
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def _improve_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict) -> Union[pd.DataFrame, dict, pd.Series]:
    if False:
        print('Hello World!')
    '\n    Improve number clusters using silh scores\n\n    :param corr_mat: (pd.DataFrame) Correlation matrix\n    :param clusters: (dict) Clusters elements\n    :param top_clusters: (dict) Improved clusters elements\n    :return: (tuple) [ordered correlation matrix, clusters, silh scores]\n    '
    pass

def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int=10, repeat: int=10) -> Union[pd.DataFrame, dict, pd.Series]:
    if False:
        print('Hello World!')
    '\n    Initial clustering step using KMeans.\n\n    :param corr_mat: (pd.DataFrame) Correlation matrix\n    :param max_num_clusters: (int) Maximum number of clusters to search for.\n    :param repeat: (int) Number of clustering algorithm repetitions.\n    :return: (tuple) [ordered correlation matrix, clusters, silh scores]\n    '
    pass

def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: tuple, new_cluster: tuple) -> tuple:
    if False:
        i = 10
        return i + 15
    '\n    Checks cluster improvement condition based on t-statistic.\n\n    :param new_tstat_mean: (float) T-statistics\n    :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement\n    :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores\n    :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores\n    :return: (tuple) Cluster\n    '
    pass

def cluster_kmeans_top(corr_mat: pd.DataFrame, repeat: int=10) -> Union[pd.DataFrame, dict, pd.Series, bool]:
    if False:
        return 10
    '\n    Improve the initial clustering by leaving clusters with high scores unchanged and modifying clusters with\n    below average scores.\n\n    :param corr_mat: (pd.DataFrame) Correlation matrix\n    :param repeat: (int) Number of clustering algorithm repetitions.\n    :return: (tuple) [correlation matrix, optimized clusters, silh scores, boolean to rerun ONC]\n    '
    pass

def get_onc_clusters(corr_mat: pd.DataFrame, repeat: int=10) -> Union[pd.DataFrame, dict, pd.Series]:
    if False:
        while True:
            i = 10
    '\n    Optimal Number of Clusters (ONC) algorithm described in the following paper:\n    `Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised\n    Learning Methods, 2015 <https://papers.ssrn.com/sol3/abstract_id=3167017>`_;\n    The code is based on the code provided by the authors of the paper.\n\n    The algorithm searches for the optimal number of clusters using the correlation matrix of elements as an input.\n\n    The correlation matrix is transformed to a matrix of distances, the K-Means algorithm is applied multiple times\n    with a different number of clusters to use. The results are evaluated on the t-statistics of the silhouette scores.\n\n    The output of the algorithm is the reordered correlation matrix (clustered elements are placed close to each other),\n    optimal clustering, and silhouette scores.\n\n    :param corr_mat: (pd.DataFrame) Correlation matrix of features\n    :param repeat: (int) Number of clustering algorithm repetitions\n    :return: (tuple) [correlation matrix, optimized clusters, silh scores]\n    '
    pass