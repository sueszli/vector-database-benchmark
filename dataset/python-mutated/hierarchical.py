import warnings
from collections import namedtuple, deque, defaultdict
from operator import attrgetter
from itertools import count
import heapq
import numpy
import scipy.cluster.hierarchy
import scipy.spatial.distance
from Orange.distance import Euclidean, PearsonR
__all__ = ['HierarchicalClustering']
_undef = object()
SINGLE = 'single'
AVERAGE = 'average'
COMPLETE = 'complete'
WEIGHTED = 'weighted'
WARD = 'ward'

def condensedform(X, mode='upper'):
    if False:
        for i in range(10):
            print('nop')
    X = numpy.asarray(X)
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1]
    N = X.shape[0]
    if mode == 'upper':
        (i, j) = numpy.triu_indices(N, k=1)
    elif mode == 'lower':
        (i, j) = numpy.tril_indices(N, k=-1)
    else:
        raise ValueError('invalid mode')
    return X[i, j]

def squareform(X, mode='upper'):
    if False:
        while True:
            i = 10
    X = numpy.asarray(X)
    k = X.shape[0]
    N = int(numpy.ceil(numpy.sqrt(k * 2)))
    assert N * (N - 1) // 2 == k
    matrix = numpy.zeros((N, N), dtype=X.dtype)
    if mode == 'upper':
        (i, j) = numpy.triu_indices(N, k=1)
        matrix[i, j] = X
        (m, n) = numpy.tril_indices(N, k=-1)
        matrix[m, n] = matrix.T[m, n]
    elif mode == 'lower':
        (i, j) = numpy.tril_indices(N, k=-1)
        matrix[i, j] = X
        (m, n) = numpy.triu_indices(N, k=1)
        matrix[m, n] = matrix.T[m, n]
    return matrix

def data_clustering(data, distance=Euclidean, linkage=AVERAGE):
    if False:
        print('Hello World!')
    "\n    Return the hierarchical clustering of the dataset's rows.\n\n    :param Orange.data.Table data: Dataset to cluster.\n    :param Orange.distance.Distance distance: A distance measure.\n    :param str linkage:\n    "
    matrix = distance(data)
    return dist_matrix_clustering(matrix, linkage=linkage)

def feature_clustering(data, distance=PearsonR, linkage=AVERAGE):
    if False:
        while True:
            i = 10
    "\n    Return the hierarchical clustering of the dataset's columns.\n\n    :param Orange.data.Table data: Dataset to cluster.\n    :param Orange.distance.Distance distance: A distance measure.\n    :param str linkage:\n    "
    matrix = distance(data, axis=0)
    return dist_matrix_clustering(matrix, linkage=linkage)

def dist_matrix_linkage(matrix, linkage=AVERAGE):
    if False:
        return 10
    '\n    Return linkage using a precomputed distance matrix.\n\n    :param Orange.misc.DistMatrix matrix:\n    :param str linkage:\n    '
    distances = condensedform(matrix)
    return scipy.cluster.hierarchy.linkage(distances, method=linkage)

def dist_matrix_clustering(matrix, linkage=AVERAGE):
    if False:
        i = 10
        return i + 15
    '\n    Return the hierarchical clustering using a precomputed distance matrix.\n\n    :param Orange.misc.DistMatrix matrix:\n    :param str linkage:\n    '
    Z = dist_matrix_linkage(matrix, linkage=linkage)
    return tree_from_linkage(Z)

def sample_clustering(X, linkage=AVERAGE, metric='euclidean'):
    if False:
        while True:
            i = 10
    assert len(X.shape) == 2
    Z = scipy.cluster.hierarchy.linkage(X, method=linkage, metric=metric)
    return tree_from_linkage(Z)

class Tree(object):
    __slots__ = ('__value', '__branches', '__hash')

    def __init__(self, value, branches=()):
        if False:
            while True:
                i = 10
        if not isinstance(branches, tuple):
            raise TypeError()
        self.__value = value
        self.__branches = branches
        self.__hash = hash((value, branches))

    def __hash__(self):
        if False:
            return 10
        return self.__hash

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, Tree) and tuple(self) == tuple(other)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Tree):
            return NotImplemented
        return tuple(self) < tuple(other)

    def __le__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Tree):
            return NotImplemented
        return tuple(self) <= tuple(other)

    def __getnewargs__(self):
        if False:
            for i in range(10):
                print('nop')
        return tuple(self)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter((self.__value, self.__branches))

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{0.__name__}(value={1!r}, branches={2!r})'.format(type(self), self.value, self.branches)

    @property
    def is_leaf(self):
        if False:
            while True:
                i = 10
        return not bool(self.branches)

    @property
    def left(self):
        if False:
            for i in range(10):
                print('nop')
        return self.branches[0]

    @property
    def right(self):
        if False:
            while True:
                i = 10
        return self.branches[-1]
    value = property(attrgetter('_Tree__value'))
    branches = property(attrgetter('_Tree__branches'))
ClusterData = namedtuple('Cluster', ['range', 'height'])
SingletonData = namedtuple('Singleton', ['range', 'height', 'index'])

class _Ranged:

    @property
    def first(self):
        if False:
            return 10
        return self.range[0]

    @property
    def last(self):
        if False:
            while True:
                i = 10
        return self.range[-1]

class ClusterData(ClusterData, _Ranged):
    __slots__ = ()

class SingletonData(SingletonData, _Ranged):
    __slots__ = ()

def tree_from_linkage(linkage):
    if False:
        return 10
    '\n    Return a Tree representation of a clustering encoded in a linkage matrix.\n\n    .. seealso:: scipy.cluster.hierarchy.linkage\n\n    '
    scipy.cluster.hierarchy.is_valid_linkage(linkage, throw=True, name='linkage')
    T = {}
    (N, _) = linkage.shape
    N = N + 1
    order = []
    for (i, (c1, c2, d, _)) in enumerate(linkage):
        if c1 < N:
            left = Tree(SingletonData(range=(len(order), len(order) + 1), height=0.0, index=int(c1)), ())
            order.append(c1)
        else:
            left = T[c1]
        if c2 < N:
            right = Tree(SingletonData(range=(len(order), len(order) + 1), height=0.0, index=int(c2)), ())
            order.append(c2)
        else:
            right = T[c2]
        t = Tree(ClusterData(range=(left.value.first, right.value.last), height=d), (left, right))
        T[N + i] = t
    root = T[N + N - 2]
    T = {}
    leaf_idx = 0
    for node in postorder(root):
        if node.is_leaf:
            T[node] = Tree(node.value._replace(range=(leaf_idx, leaf_idx + 1)), ())
            leaf_idx += 1
        else:
            (left, right) = (T[node.left].value, T[node.right].value)
            assert left.first < right.first
            t = Tree(node.value._replace(range=(left.range[0], right.range[1])), tuple((T[ch] for ch in node.branches)))
            assert t.value.range[0] <= t.value.range[-1]
            assert left.first == t.value.first and right.last == t.value.last
            assert t.value.first < right.first
            assert t.value.last > left.last
            T[node] = t
    return T[root]

def linkage_from_tree(tree: Tree) -> numpy.ndarray:
    if False:
        print('Hello World!')
    leafs = [n for n in preorder(tree) if n.is_leaf]
    Z = numpy.zeros((len(leafs) - 1, 4), float)
    i = 0
    node_to_i = defaultdict(count(len(leafs)).__next__)
    for node in postorder(tree):
        if node.is_leaf:
            node_to_i[node] = node.value.index
        else:
            assert len(node.branches) == 2
            assert node.left in node_to_i
            assert node.right in node_to_i
            Z[i] = [node_to_i[node.left], node_to_i[node.right], node.value.height, 0]
            _ni = node_to_i[node]
            assert _ni == Z.shape[0] + i + 1
            i += 1
    assert i == Z.shape[0]
    return Z

def postorder(tree, branches=attrgetter('branches')):
    if False:
        print('Hello World!')
    stack = deque([tree])
    visited = set()
    while stack:
        current = stack.popleft()
        children = branches(current)
        if children:
            if current in visited:
                yield current
            else:
                stack.extendleft([current])
                stack.extendleft(reversed(children))
                visited.add(current)
        else:
            yield current
            visited.add(current)

def preorder(tree, branches=attrgetter('branches')):
    if False:
        print('Hello World!')
    stack = deque([tree])
    while stack:
        current = stack.popleft()
        yield current
        children = branches(current)
        if children:
            stack.extendleft(reversed(children))

def leaves(tree, branches=attrgetter('branches')):
    if False:
        return 10
    '\n    Return an iterator over the leaf nodes in a tree structure.\n    '
    return (node for node in postorder(tree, branches) if node.is_leaf)

def prune(cluster, level=None, height=None, condition=None):
    if False:
        while True:
            i = 10
    '\n    Prune the clustering instance ``cluster``.\n\n    :param Tree cluster: Cluster root node to prune.\n    :param int level: If not `None` prune all clusters deeper then `level`.\n    :param float height:\n        If not `None` prune all clusters with height lower then `height`.\n    :param function condition:\n        If not `None condition must be a `Tree -> bool` function\n        evaluating to `True` if the cluster should be pruned.\n\n    .. note::\n        At least one `level`, `height` or `condition` argument needs to\n        be supplied.\n\n    '
    if not any((arg is not None for arg in [level, height, condition])):
        raise ValueError('At least one pruning argument must be supplied')
    level_check = height_check = condition_check = lambda cl: False
    if level is not None:
        cluster_depth = cluster_depths(cluster)
        level_check = lambda cl: cluster_depth[cl] >= level
    if height is not None:
        height_check = lambda cl: cl.value.height <= height
    if condition is not None:
        condition_check = condition

    def check_all(cl):
        if False:
            while True:
                i = 10
        return level_check(cl) or height_check(cl) or condition_check(cl)
    T = {}
    for node in postorder(cluster):
        if check_all(node):
            if node.is_leaf:
                T[node] = node
            else:
                T[node] = Tree(node.value, ())
        else:
            T[node] = Tree(node.value, tuple((T[ch] for ch in node.branches)))
    return T[cluster]

def cluster_depths(cluster):
    if False:
        return 10
    '\n    Return a dictionary mapping :class:`Tree` instances to their depth.\n\n    :param Tree cluster: Root cluster\n    :rtype: class:`dict`\n\n    '
    depths = {}
    depths[cluster] = 0
    for cluster in preorder(cluster):
        cl_depth = depths[cluster]
        depths.update(dict.fromkeys(cluster.branches, cl_depth + 1))
    return depths

def top_clusters(tree, k):
    if False:
        return 10
    '\n    Return `k` topmost clusters from hierarchical clustering.\n\n    :param Tree root: Root cluster.\n    :param int k: Number of top clusters.\n\n    :rtype: list of :class:`Tree` instances\n    '

    def item(node):
        if False:
            return 10
        return ((node.is_leaf, -node.value.height), node)
    heap = [item(tree)]
    while len(heap) < k:
        (_, cl) = heap[0]
        if cl.is_leaf:
            assert all((n.is_leaf for (_, n) in heap))
            break
        (key, cl) = heapq.heappop(heap)
        (left, right) = (cl.left, cl.right)
        heapq.heappush(heap, item(left))
        heapq.heappush(heap, item(right))
    return [n for (_, n) in heap]

def optimal_leaf_ordering(tree: Tree, distances: numpy.ndarray, progress_callback=_undef) -> Tree:
    if False:
        i = 10
        return i + 15
    '\n    Order the leaves in the clustering tree.\n\n    :param Tree tree:\n        Binary hierarchical clustering tree.\n    :param numpy.ndarray distances:\n        A (N, N) numpy.ndarray of distances that were used to compute\n        the clustering.\n\n    .. seealso:: scipy.cluster.hierarchy.optimal_leaf_ordering\n    '
    if progress_callback is not _undef:
        warnings.warn("'progress_callback' parameter is deprecated and ignored. Passing it will raise an error in the future.", FutureWarning, stacklevel=2)
    Z = linkage_from_tree(tree)
    y = condensedform(numpy.asarray(distances))
    Zopt = scipy.cluster.hierarchy.optimal_leaf_ordering(Z, y)
    return tree_from_linkage(Zopt)

class HierarchicalClustering:

    def __init__(self, n_clusters=2, linkage=AVERAGE):
        if False:
            return 10
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        if False:
            while True:
                i = 10
        self.tree = dist_matrix_clustering(X, linkage=self.linkage)
        cut = top_clusters(self.tree, self.n_clusters)
        labels = numpy.zeros(self.tree.value.last)
        for (i, cl) in enumerate(cut):
            indices = [leaf.value.index for leaf in leaves(cl)]
            labels[indices] = i
        self.labels = labels

    def fit_predict(self, X, y=None):
        if False:
            return 10
        self.fit(X)
        return self.labels