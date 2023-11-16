import networkx as nx
import numpy as np
from scipy.sparse import linalg
from .._shared.utils import deprecate_kwarg
from . import _ncut, _ncut_cy

def cut_threshold(labels, rag, thresh, in_place=True):
    if False:
        i = 10
        return i + 15
    'Combine regions separated by weight less than threshold.\n\n    Given an image\'s labels and its RAG, output new labels by\n    combining regions whose nodes are separated by a weight less\n    than the given threshold.\n\n    Parameters\n    ----------\n    labels : ndarray\n        The array of labels.\n    rag : RAG\n        The region adjacency graph.\n    thresh : float\n        The threshold. Regions connected by edges with smaller weights are\n        combined.\n    in_place : bool\n        If set, modifies `rag` in place. The function will remove the edges\n        with weights less that `thresh`. If set to `False` the function\n        makes a copy of `rag` before proceeding.\n\n    Returns\n    -------\n    out : ndarray\n        The new labelled array.\n\n    Examples\n    --------\n    >>> from skimage import data, segmentation, graph\n    >>> img = data.astronaut()\n    >>> labels = segmentation.slic(img)\n    >>> rag = graph.rag_mean_color(img, labels)\n    >>> new_labels = graph.cut_threshold(labels, rag, 10)\n\n    References\n    ----------\n    .. [1] Alain Tremeau and Philippe Colantoni\n           "Regions Adjacency Graph Applied To Color Image Segmentation"\n           :DOI:`10.1109/83.841950`\n\n    '
    if not in_place:
        rag = rag.copy()
    to_remove = [(x, y) for (x, y, d) in rag.edges(data=True) if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)
    comps = nx.connected_components(rag)
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for (i, nodes) in enumerate(comps):
        for node in nodes:
            for label in rag.nodes[node]['labels']:
                map_array[label] = i
    return map_array[labels]

@deprecate_kwarg({'random_state': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def cut_normalized(labels, rag, thresh=0.001, num_cuts=10, in_place=True, max_edge=1.0, *, rng=None):
    if False:
        for i in range(10):
            print('nop')
    'Perform Normalized Graph cut on the Region Adjacency Graph.\n\n    Given an image\'s labels and its similarity RAG, recursively perform\n    a 2-way normalized cut on it. All nodes belonging to a subgraph\n    that cannot be cut further are assigned a unique label in the\n    output.\n\n    Parameters\n    ----------\n    labels : ndarray\n        The array of labels.\n    rag : RAG\n        The region adjacency graph.\n    thresh : float\n        The threshold. A subgraph won\'t be further subdivided if the\n        value of the N-cut exceeds `thresh`.\n    num_cuts : int\n        The number or N-cuts to perform before determining the optimal one.\n    in_place : bool\n        If set, modifies `rag` in place. For each node `n` the function will\n        set a new attribute ``rag.nodes[n][\'ncut label\']``.\n    max_edge : float, optional\n        The maximum possible value of an edge in the RAG. This corresponds to\n        an edge between identical regions. This is used to put self\n        edges in the RAG.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n        The `rng` is used to determine the starting point\n        of `scipy.sparse.linalg.eigsh`.\n\n    Returns\n    -------\n    out : ndarray\n        The new labeled array.\n\n    Examples\n    --------\n    >>> from skimage import data, segmentation, graph\n    >>> img = data.astronaut()\n    >>> labels = segmentation.slic(img)\n    >>> rag = graph.rag_mean_color(img, labels, mode=\'similarity\')\n    >>> new_labels = graph.cut_normalized(labels, rag)\n\n    References\n    ----------\n    .. [1] Shi, J.; Malik, J., "Normalized cuts and image segmentation",\n           Pattern Analysis and Machine Intelligence,\n           IEEE Transactions on, vol. 22, no. 8, pp. 888-905, August 2000.\n\n    '
    rng = np.random.default_rng(rng)
    if not in_place:
        rag = rag.copy()
    for node in rag.nodes():
        rag.add_edge(node, node, weight=max_edge)
    _ncut_relabel(rag, thresh, num_cuts, rng)
    map_array = np.zeros(labels.max() + 1, dtype=labels.dtype)
    for (n, d) in rag.nodes(data=True):
        map_array[d['labels']] = d['ncut label']
    return map_array[labels]

def partition_by_cut(cut, rag):
    if False:
        for i in range(10):
            print('nop')
    'Compute resulting subgraphs from given bi-partition.\n\n    Parameters\n    ----------\n    cut : array\n        A array of booleans. Elements set to `True` belong to one\n        set.\n    rag : RAG\n        The Region Adjacency Graph.\n\n    Returns\n    -------\n    sub1, sub2 : RAG\n        The two resulting subgraphs from the bi-partition.\n    '
    nodes1 = [n for (i, n) in enumerate(rag.nodes()) if cut[i]]
    nodes2 = [n for (i, n) in enumerate(rag.nodes()) if not cut[i]]
    sub1 = rag.subgraph(nodes1)
    sub2 = rag.subgraph(nodes2)
    return (sub1, sub2)

def get_min_ncut(ev, d, w, num_cuts):
    if False:
        while True:
            i = 10
    'Threshold an eigenvector evenly, to determine minimum ncut.\n\n    Parameters\n    ----------\n    ev : array\n        The eigenvector to threshold.\n    d : ndarray\n        The diagonal matrix of the graph.\n    w : ndarray\n        The weight matrix of the graph.\n    num_cuts : int\n        The number of evenly spaced thresholds to check for.\n\n    Returns\n    -------\n    mask : array\n        The array of booleans which denotes the bi-partition.\n    mcut : float\n        The value of the minimum ncut.\n    '
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return (min_mask, mcut)
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = _ncut.ncut_cost(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost
    return (min_mask, mcut)

def _label_all(rag, attr_name):
    if False:
        while True:
            i = 10
    'Assign a unique integer to the given attribute in the RAG.\n\n    This function assumes that all labels in `rag` are unique. It\n    picks up a random label from them and assigns it to the `attr_name`\n    attribute of all the nodes.\n\n    rag : RAG\n        The Region Adjacency Graph.\n    attr_name : string\n        The attribute to which a unique integer is assigned.\n    '
    node = min(rag.nodes())
    new_label = rag.nodes[node]['labels'][0]
    for (n, d) in rag.nodes(data=True):
        d[attr_name] = new_label

def _ncut_relabel(rag, thresh, num_cuts, random_generator):
    if False:
        print('Hello World!')
    "Perform Normalized Graph cut on the Region Adjacency Graph.\n\n    Recursively partition the graph into 2, until further subdivision\n    yields a cut greater than `thresh` or such a cut cannot be computed.\n    For such a subgraph, indices to labels of all its nodes map to a single\n    unique value.\n\n    Parameters\n    ----------\n    rag : RAG\n        The region adjacency graph.\n    thresh : float\n        The threshold. A subgraph won't be further subdivided if the\n        value of the N-cut exceeds `thresh`.\n    num_cuts : int\n        The number or N-cuts to perform before determining the optimal one.\n    random_generator : `numpy.random.Generator`\n        Provides initial values for eigenvalue solver.\n    "
    (d, w) = _ncut.DW_matrices(rag)
    m = w.shape[0]
    if m > 2:
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        A = d2 * (d - w) * d2
        v0 = random_generator.random(A.shape[0])
        (vals, vectors) = linalg.eigsh(A, which='SM', v0=v0, k=min(100, m - 2))
        (vals, vectors) = (np.real(vals), np.real(vectors))
        index2 = _ncut_cy.argmin2(vals)
        ev = vectors[:, index2]
        (cut_mask, mcut) = get_min_ncut(ev, d, w, num_cuts)
        if mcut < thresh:
            (sub1, sub2) = partition_by_cut(cut_mask, rag)
            _ncut_relabel(sub1, thresh, num_cuts, random_generator)
            _ncut_relabel(sub2, thresh, num_cuts, random_generator)
            return
    _label_all(rag, 'ncut label')