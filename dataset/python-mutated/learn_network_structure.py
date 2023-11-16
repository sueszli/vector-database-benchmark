__author__ = 'fs'
import numpy as np
from sklearn import cluster, covariance, manifold

def learn_network_structure(ts_returns_data, names, alphas=4, cv=5, mode='cd', assume_centered=False, n_components=2, n_neighbors=5, eigen_solver='dense', method='standard', neighbors_algorithm='auto', random_state=None, n_jobs=None, standardise=False):
    if False:
        while True:
            i = 10
    '\n\n\tParameters\n\t----------\n\tts_returns_data : array-like of shape [n_samples, n_instruments]\n\t                  time series matrix of returns\n\n\tnames : array-like of shape [n_samples, 1]\n\t        Individual names of the financial instrument\n\n\talphas : int or positive float, optional\n\t         Number of points on the grids to be used\n\n\tcv : int, optional\n\t     Number of folds for cross-validation splitting strategy\n\n\tmode : str, optional\n\t       Solver to use to compute the graph\n\n\tassume_centered : bool, optional\n                      Centre the data if False.\n\n\tn_components : int\n\t               Number of components for the manifold\n\n\tn_neighbors: int\n                 Number of neighbours to consider for each point\n\n\teigen_solver : str\n\t               Algorithm to compute eigenvalues\n\n\tmethod : str\n             Algorithm to use for local linear embedding\n\tneighbors_algorithm : str\n\t                      Algorithm to use for nearest neighbours search\n\n\trandom_state : int, RandomState instance or None, optional\n\t               If int, random_state is the seed used by the random number generator.\n\t               If RandomState instance, random_state is the random number generator.\n\t               If None, the random number generator is the RandomState instance used by np.random.\n\t               Used when eigen_solver == ‘arpack’\n\n\tn_jobs : int or None, optional\n\t         number of parallel jobs to run\n\n\tstandardise : bool\n\t              standardise data if True\n\n\tReturns : sklearn.covariance.graph_lasso_.GraphicalLassoCV\n\n              sklearn.manifold.locally_linear.LocallyLinearEmbedding\n\n              array-like of shape [n_components, n_instruments]\n              Transformed embedding vectors\n\n              array-like of shape [n_instruments, 1]\n              numeric identifier of each cluster\n\n\n\n\t-------\n\t'
    if not isinstance(ts_returns_data, (np.ndarray, np.generic)):
        raise TypeError('ts_returns_data must be of class ndarray')
    edge_model = covariance.GraphicalLassoCV(alphas=alphas, cv=cv, mode=mode, assume_centered=assume_centered)
    edge_model.fit(ts_returns_data)
    (_, labels) = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()
    for i in range(n_labels + 1):
        print('Cluster %i: %s' % (i + 1, ', '.join(names[labels == i])))
    node_position_model = manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver=eigen_solver, n_neighbors=n_neighbors, method=method, neighbors_algorithm=neighbors_algorithm, random_state=random_state, n_jobs=n_jobs)
    embedding = node_position_model.fit_transform(ts_returns_data.T).T
    if standardise:
        standard_ret = ts_returns_data.copy()
        standard_ret /= ts_returns_data.std(axis=0)
        edge_model.fit(standard_ret)
        (_, labels) = cluster.affinity_propagation(edge_model.covariance_)
        n_labels = labels.max()
        for i in range(n_labels + 1):
            print('Cluster %i: %s' % (i + 1, ', '.join(names[labels == i])))
        node_position_model = manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver=eigen_solver, n_neighbors=n_neighbors, method=method, neighbors_algorithm=neighbors_algorithm, random_state=random_state, n_jobs=n_jobs)
        embedding = node_position_model.fit_transform(ts_returns_data.T).T
    return (edge_model, node_position_model, embedding, labels)