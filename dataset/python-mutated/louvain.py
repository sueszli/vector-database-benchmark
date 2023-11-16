"""Python port for Louvain clustering, available at
https://github.com/taynaud/python-louvain

Original C++ implementation available at
https://sites.google.com/site/findcommunities/
"""
import numpy as np
import networkx as nx
from community import best_partition
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from Orange.clustering.clustering import Clustering
from Orange.data import Table
__all__ = ['Louvain', 'matrix_to_knn_graph']

def jaccard(x, y):
    if False:
        while True:
            i = 10
    'Compute the Jaccard similarity between two sets.'
    return len(x & y) / len(x | y)

def matrix_to_knn_graph(data, k_neighbors, metric, progress_callback=None):
    if False:
        while True:
            i = 10
    'Convert data matrix to a graph using a nearest neighbors approach with\n    the Jaccard similarity as the edge weights.\n\n    Parameters\n    ----------\n    data : np.ndarray\n    k_neighbors : int\n    metric : str\n        A distance metric supported by sklearn.\n    progress_callback : Callable[[float], None]\n\n    Returns\n    -------\n    nx.Graph\n\n    '
    if metric == 'cosine':
        data = data / np.linalg.norm(data, axis=1)[:, None]
        metric = 'euclidean'
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric).fit(data)
    nearest_neighbors = knn.kneighbors(data, return_distance=False)
    nearest_neighbors = list(map(set, nearest_neighbors))
    num_nodes = len(nearest_neighbors)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(data)))
    for (idx, node) in enumerate(graph.nodes):
        if progress_callback:
            progress_callback(idx / num_nodes)
        for neighbor in nearest_neighbors[node]:
            graph.add_edge(node, neighbor, weight=jaccard(nearest_neighbors[node], nearest_neighbors[neighbor]))
    return graph

class LouvainMethod(BaseEstimator):

    def __init__(self, k_neighbors=30, metric='l2', resolution=1.0, random_state=None):
        if False:
            for i in range(10):
                print('nop')
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.resolution = resolution
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        if False:
            return 10
        graph = matrix_to_knn_graph(X, metric=self.metric, k_neighbors=self.k_neighbors)
        return self.fit_graph(graph)

    def fit_graph(self, graph):
        if False:
            while True:
                i = 10
        partition = best_partition(graph, resolution=self.resolution, random_state=self.random_state)
        self.labels_ = np.fromiter(list(zip(*sorted(partition.items())))[1], dtype=int)
        return self

class Louvain(Clustering):
    """Louvain clustering for community detection in graphs.

    Louvain clustering is a community detection algorithm for detecting
    clusters of "communities" in graphs. As such, tabular data must first
    be converted into graph form. This is typically done by computing the
    KNN graph on the input data.

    Attributes
    ----------
    k_neighbors : Optional[int]
        The number of nearest neighbors to use for the KNN graph if
        tabular data is passed.

    metric : Optional[str]
        The metric to use to compute the nearest neighbors.

    resolution : Optional[float]
        The resolution is a parameter of the Louvain method that affects
        the size of the recovered clusters.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.
    """
    __wraps__ = LouvainMethod

    def __init__(self, k_neighbors=30, metric='l2', resolution=1.0, random_state=None, preprocessors=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(preprocessors, vars())

    def get_model(self, data):
        if False:
            while True:
                i = 10
        if isinstance(data, nx.Graph):
            return self.__returns__(self.__wraps__(**self.params).fit_graph(data))
        else:
            return super().get_model(data)
if __name__ == '__main__':
    d = Table('iris')
    louvain = Louvain(5)
    clusters = louvain(d)