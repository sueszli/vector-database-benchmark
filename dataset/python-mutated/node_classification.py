""" This module provides the functions for node classification problem.

The functions in this module are not imported
into the top level `networkx` namespace.
You can access these functions by importing
the `networkx.algorithms.node_classification` modules,
then accessing the functions as attributes of `node_classification`.
For example:

  >>> from networkx.algorithms import node_classification
  >>> G = nx.path_graph(4)
  >>> G.edges()
  EdgeView([(0, 1), (1, 2), (2, 3)])
  >>> G.nodes[0]["label"] = "A"
  >>> G.nodes[3]["label"] = "B"
  >>> node_classification.harmonic_function(G)
  ['A', 'A', 'B', 'B']

References
----------
Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
Semi-supervised learning using gaussian fields and harmonic functions.
In ICML (Vol. 3, pp. 912-919).
"""
import networkx as nx
__all__ = ['harmonic_function', 'local_and_global_consistency']

@nx.utils.not_implemented_for('directed')
@nx._dispatch(node_attrs='label_name')
def harmonic_function(G, max_iter=30, label_name='label'):
    if False:
        print('Hello World!')
    'Node classification by Harmonic function\n\n    Function for computing Harmonic function algorithm by Zhu et al.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n    max_iter : int\n        maximum number of iterations allowed\n    label_name : string\n        name of target labels to predict\n\n    Returns\n    -------\n    predicted : list\n        List of length ``len(G)`` with the predicted labels for each node.\n\n    Raises\n    ------\n    NetworkXError\n        If no nodes in `G` have attribute `label_name`.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import node_classification\n    >>> G = nx.path_graph(4)\n    >>> G.nodes[0]["label"] = "A"\n    >>> G.nodes[3]["label"] = "B"\n    >>> G.nodes(data=True)\n    NodeDataView({0: {\'label\': \'A\'}, 1: {}, 2: {}, 3: {\'label\': \'B\'}})\n    >>> G.edges()\n    EdgeView([(0, 1), (1, 2), (2, 3)])\n    >>> predicted = node_classification.harmonic_function(G)\n    >>> predicted\n    [\'A\', \'A\', \'B\', \'B\']\n\n    References\n    ----------\n    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).\n    Semi-supervised learning using gaussian fields and harmonic functions.\n    In ICML (Vol. 3, pp. 912-919).\n    '
    import numpy as np
    import scipy as sp
    X = nx.to_scipy_sparse_array(G)
    (labels, label_dict) = _get_label_info(G, label_name)
    if labels.shape[0] == 0:
        raise nx.NetworkXError(f"No node on the input graph is labeled by '{label_name}'.")
    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = np.zeros((n_samples, n_classes))
    degrees = X.sum(axis=0)
    degrees[degrees == 0] = 1
    D = sp.sparse.csr_array(sp.sparse.diags(1.0 / degrees, offsets=0))
    P = (D @ X).tolil()
    P[labels[:, 0]] = 0
    B = np.zeros((n_samples, n_classes))
    B[labels[:, 0], labels[:, 1]] = 1
    for _ in range(max_iter):
        F = P @ F + B
    return label_dict[np.argmax(F, axis=1)].tolist()

@nx.utils.not_implemented_for('directed')
@nx._dispatch(node_attrs='label_name')
def local_and_global_consistency(G, alpha=0.99, max_iter=30, label_name='label'):
    if False:
        i = 10
        return i + 15
    'Node classification by Local and Global Consistency\n\n    Function for computing Local and global consistency algorithm by Zhou et al.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n    alpha : float\n        Clamping factor\n    max_iter : int\n        Maximum number of iterations allowed\n    label_name : string\n        Name of target labels to predict\n\n    Returns\n    -------\n    predicted : list\n        List of length ``len(G)`` with the predicted labels for each node.\n\n    Raises\n    ------\n    NetworkXError\n        If no nodes in `G` have attribute `label_name`.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import node_classification\n    >>> G = nx.path_graph(4)\n    >>> G.nodes[0]["label"] = "A"\n    >>> G.nodes[3]["label"] = "B"\n    >>> G.nodes(data=True)\n    NodeDataView({0: {\'label\': \'A\'}, 1: {}, 2: {}, 3: {\'label\': \'B\'}})\n    >>> G.edges()\n    EdgeView([(0, 1), (1, 2), (2, 3)])\n    >>> predicted = node_classification.local_and_global_consistency(G)\n    >>> predicted\n    [\'A\', \'A\', \'B\', \'B\']\n\n    References\n    ----------\n    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).\n    Learning with local and global consistency.\n    Advances in neural information processing systems, 16(16), 321-328.\n    '
    import numpy as np
    import scipy as sp
    X = nx.to_scipy_sparse_array(G)
    (labels, label_dict) = _get_label_info(G, label_name)
    if labels.shape[0] == 0:
        raise nx.NetworkXError(f"No node on the input graph is labeled by '{label_name}'.")
    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = np.zeros((n_samples, n_classes))
    degrees = X.sum(axis=0)
    degrees[degrees == 0] = 1
    D2 = np.sqrt(sp.sparse.csr_array(sp.sparse.diags(1.0 / degrees, offsets=0)))
    P = alpha * (D2 @ X @ D2)
    B = np.zeros((n_samples, n_classes))
    B[labels[:, 0], labels[:, 1]] = 1 - alpha
    for _ in range(max_iter):
        F = P @ F + B
    return label_dict[np.argmax(F, axis=1)].tolist()

def _get_label_info(G, label_name):
    if False:
        return 10
    'Get and return information of labels from the input graph\n\n    Parameters\n    ----------\n    G : Network X graph\n    label_name : string\n        Name of the target label\n\n    Returns\n    -------\n    labels : numpy array, shape = [n_labeled_samples, 2]\n        Array of pairs of labeled node ID and label ID\n    label_dict : numpy array, shape = [n_classes]\n        Array of labels\n        i-th element contains the label corresponding label ID `i`\n    '
    import numpy as np
    labels = []
    label_to_id = {}
    lid = 0
    for (i, n) in enumerate(G.nodes(data=True)):
        if label_name in n[1]:
            label = n[1][label_name]
            if label not in label_to_id:
                label_to_id[label] = lid
                lid += 1
            labels.append([i, label_to_id[label]])
    labels = np.array(labels)
    label_dict = np.array([label for (label, _) in sorted(label_to_id.items(), key=lambda x: x[1])])
    return (labels, label_dict)