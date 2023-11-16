import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
__all__ = ['minkowski_distance_p', 'minkowski_distance', 'distance_matrix', 'Rectangle', 'KDTree']

def minkowski_distance_p(x, y, p=2):
    if False:
        i = 10
        return i + 15
    'Compute the pth power of the L**p distance between two arrays.\n\n    For efficiency, this function computes the L**p distance but does\n    not extract the pth root. If `p` is 1 or infinity, this is equal to\n    the actual L**p distance.\n\n    The last dimensions of `x` and `y` must be the same length.  Any\n    other dimensions must be compatible for broadcasting.\n\n    Parameters\n    ----------\n    x : (..., K) array_like\n        Input array.\n    y : (..., K) array_like\n        Input array.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n\n    Returns\n    -------\n    dist : ndarray\n        pth power of the distance between the input arrays.\n\n    Examples\n    --------\n    >>> from scipy.spatial import minkowski_distance_p\n    >>> minkowski_distance_p([[0, 0], [0, 0]], [[1, 1], [0, 1]])\n    array([2, 1])\n\n    '
    x = np.asarray(x)
    y = np.asarray(y)
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)
    if p == np.inf:
        return np.amax(np.abs(y - x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis=-1)
    else:
        return np.sum(np.abs(y - x) ** p, axis=-1)

def minkowski_distance(x, y, p=2):
    if False:
        i = 10
        return i + 15
    'Compute the L**p distance between two arrays.\n\n    The last dimensions of `x` and `y` must be the same length.  Any\n    other dimensions must be compatible for broadcasting.\n\n    Parameters\n    ----------\n    x : (..., K) array_like\n        Input array.\n    y : (..., K) array_like\n        Input array.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n\n    Returns\n    -------\n    dist : ndarray\n        Distance between the input arrays.\n\n    Examples\n    --------\n    >>> from scipy.spatial import minkowski_distance\n    >>> minkowski_distance([[0, 0], [0, 0]], [[1, 1], [0, 1]])\n    array([ 1.41421356,  1.        ])\n\n    '
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p) ** (1.0 / p)

class Rectangle:
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """

    def __init__(self, maxes, mins):
        if False:
            for i in range(10):
                print('nop')
        'Construct a hyperrectangle.'
        self.maxes = np.maximum(maxes, mins).astype(float)
        self.mins = np.minimum(maxes, mins).astype(float)
        (self.m,) = self.maxes.shape

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Rectangle %s>' % list(zip(self.mins, self.maxes))

    def volume(self):
        if False:
            for i in range(10):
                print('nop')
        'Total volume.'
        return np.prod(self.maxes - self.mins)

    def split(self, d, split):
        if False:
            return 10
        'Produce two hyperrectangles by splitting.\n\n        In general, if you need to compute maximum and minimum\n        distances to the children, it can be done more efficiently\n        by updating the maximum and minimum distances to the parent.\n\n        Parameters\n        ----------\n        d : int\n            Axis to split hyperrectangle along.\n        split : float\n            Position along axis `d` to split at.\n\n        '
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return (less, greater)

    def min_distance_point(self, x, p=2.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the minimum distance between input and points in the\n        hyperrectangle.\n\n        Parameters\n        ----------\n        x : array_like\n            Input.\n        p : float, optional\n            Input.\n\n        '
        return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - x, x - self.maxes)), p)

    def max_distance_point(self, x, p=2.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the maximum distance between input and points in the hyperrectangle.\n\n        Parameters\n        ----------\n        x : array_like\n            Input array.\n        p : float, optional\n            Input.\n\n        '
        return minkowski_distance(0, np.maximum(self.maxes - x, x - self.mins), p)

    def min_distance_rectangle(self, other, p=2.0):
        if False:
            i = 10
            return i + 15
        '\n        Compute the minimum distance between points in the two hyperrectangles.\n\n        Parameters\n        ----------\n        other : hyperrectangle\n            Input.\n        p : float\n            Input.\n\n        '
        return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - other.maxes, other.mins - self.maxes)), p)

    def max_distance_rectangle(self, other, p=2.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the maximum distance between points in the two hyperrectangles.\n\n        Parameters\n        ----------\n        other : hyperrectangle\n            Input.\n        p : float, optional\n            Input.\n\n        '
        return minkowski_distance(0, np.maximum(self.maxes - other.mins, other.maxes - self.mins), p)

class KDTree(cKDTree):
    """kd-tree for quick nearest-neighbor lookup.

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Default: 10.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with `copy_data=True`.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    size : int
        The number of nodes in the tree.

    """

    class node:

        @staticmethod
        def _create(ckdtree_node=None):
            if False:
                i = 10
                return i + 15
            'Create either an inner or leaf node, wrapping a cKDTreeNode instance'
            if ckdtree_node is None:
                return KDTree.node(ckdtree_node)
            elif ckdtree_node.split_dim == -1:
                return KDTree.leafnode(ckdtree_node)
            else:
                return KDTree.innernode(ckdtree_node)

        def __init__(self, ckdtree_node=None):
            if False:
                for i in range(10):
                    print('nop')
            if ckdtree_node is None:
                ckdtree_node = cKDTreeNode()
            self._node = ckdtree_node

        def __lt__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            return id(self) < id(other)

        def __gt__(self, other):
            if False:
                while True:
                    i = 10
            return id(self) > id(other)

        def __le__(self, other):
            if False:
                while True:
                    i = 10
            return id(self) <= id(other)

        def __ge__(self, other):
            if False:
                i = 10
                return i + 15
            return id(self) >= id(other)

        def __eq__(self, other):
            if False:
                i = 10
                return i + 15
            return id(self) == id(other)

    class leafnode(node):

        @property
        def idx(self):
            if False:
                print('Hello World!')
            return self._node.indices

        @property
        def children(self):
            if False:
                return 10
            return self._node.children

    class innernode(node):

        def __init__(self, ckdtreenode):
            if False:
                return 10
            assert isinstance(ckdtreenode, cKDTreeNode)
            super().__init__(ckdtreenode)
            self.less = KDTree.node._create(ckdtreenode.lesser)
            self.greater = KDTree.node._create(ckdtreenode.greater)

        @property
        def split_dim(self):
            if False:
                return 10
            return self._node.split_dim

        @property
        def split(self):
            if False:
                print('Hello World!')
            return self._node.split

        @property
        def children(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._node.children

    @property
    def tree(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_tree'):
            self._tree = KDTree.node._create(super().tree)
        return self._tree

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None):
        if False:
            i = 10
            return i + 15
        data = np.asarray(data)
        if data.dtype.kind == 'c':
            raise TypeError('KDTree does not work with complex data')
        super().__init__(data, leafsize, compact_nodes, copy_data, balanced_tree, boxsize)

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        if False:
            return 10
        'Query the kd-tree for nearest neighbors.\n\n        Parameters\n        ----------\n        x : array_like, last dimension self.m\n            An array of points to query.\n        k : int or Sequence[int], optional\n            Either the number of nearest neighbors to return, or a list of the\n            k-th nearest neighbors to return, starting from 1.\n        eps : nonnegative float, optional\n            Return approximate nearest neighbors; the kth returned value\n            is guaranteed to be no further than (1+eps) times the\n            distance to the real kth nearest neighbor.\n        p : float, 1<=p<=infinity, optional\n            Which Minkowski p-norm to use.\n            1 is the sum-of-absolute-values distance ("Manhattan" distance).\n            2 is the usual Euclidean distance.\n            infinity is the maximum-coordinate-difference distance.\n            A large, finite p may cause a ValueError if overflow can occur.\n        distance_upper_bound : nonnegative float, optional\n            Return only neighbors within this distance. This is used to prune\n            tree searches, so if you are doing a series of nearest-neighbor\n            queries, it may help to supply the distance to the nearest neighbor\n            of the most recent point.\n        workers : int, optional\n            Number of workers to use for parallel processing. If -1 is given\n            all CPU threads are used. Default: 1.\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        d : float or array of floats\n            The distances to the nearest neighbors.\n            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape\n            ``tuple+(k,)``.\n            When k == 1, the last dimension of the output is squeezed.\n            Missing neighbors are indicated with infinite distances.\n            Hits are sorted by distance (nearest first).\n\n            .. versionchanged:: 1.9.0\n               Previously if ``k=None``, then `d` was an object array of\n               shape ``tuple``, containing lists of distances. This behavior\n               has been removed, use `query_ball_point` instead.\n\n        i : integer or array of integers\n            The index of each neighbor in ``self.data``.\n            ``i`` is the same shape as d.\n            Missing neighbors are indicated with ``self.n``.\n\n        Examples\n        --------\n\n        >>> import numpy as np\n        >>> from scipy.spatial import KDTree\n        >>> x, y = np.mgrid[0:5, 2:8]\n        >>> tree = KDTree(np.c_[x.ravel(), y.ravel()])\n\n        To query the nearest neighbours and return squeezed result, use\n\n        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)\n        >>> print(dd, ii, sep=\'\\n\')\n        [2.         0.2236068]\n        [ 0 13]\n\n        To query the nearest neighbours and return unsqueezed result, use\n\n        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])\n        >>> print(dd, ii, sep=\'\\n\')\n        [[2.        ]\n         [0.2236068]]\n        [[ 0]\n         [13]]\n\n        To query the second nearest neighbours and return unsqueezed result,\n        use\n\n        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])\n        >>> print(dd, ii, sep=\'\\n\')\n        [[2.23606798]\n         [0.80622577]]\n        [[ 6]\n         [19]]\n\n        To query the first and second nearest neighbours, use\n\n        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)\n        >>> print(dd, ii, sep=\'\\n\')\n        [[2.         2.23606798]\n         [0.2236068  0.80622577]]\n        [[ 0  6]\n         [13 19]]\n\n        or, be more specific\n\n        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])\n        >>> print(dd, ii, sep=\'\\n\')\n        [[2.         2.23606798]\n         [0.2236068  0.80622577]]\n        [[ 0  6]\n         [13 19]]\n\n        '
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError('KDTree does not work with complex data')
        if k is None:
            raise ValueError('k must be an integer or a sequence of integers')
        (d, i) = super().query(x, k, eps, p, distance_upper_bound, workers)
        if isinstance(i, int):
            i = np.intp(i)
        return (d, i)

    def query_ball_point(self, x, r, p=2.0, eps=0, workers=1, return_sorted=None, return_length=False):
        if False:
            print('Hello World!')
        "Find all points within distance r of point(s) x.\n\n        Parameters\n        ----------\n        x : array_like, shape tuple + (self.m,)\n            The point or points to search for neighbors of.\n        r : array_like, float\n            The radius of points to return, must broadcast to the length of x.\n        p : float, optional\n            Which Minkowski p-norm to use.  Should be in the range [1, inf].\n            A finite large p may cause a ValueError if overflow can occur.\n        eps : nonnegative float, optional\n            Approximate search. Branches of the tree are not explored if their\n            nearest points are further than ``r / (1 + eps)``, and branches are\n            added in bulk if their furthest points are nearer than\n            ``r * (1 + eps)``.\n        workers : int, optional\n            Number of jobs to schedule for parallel processing. If -1 is given\n            all processors are used. Default: 1.\n\n            .. versionadded:: 1.6.0\n        return_sorted : bool, optional\n            Sorts returned indicies if True and does not sort them if False. If\n            None, does not sort single point queries, but does sort\n            multi-point queries which was the behavior before this option\n            was added.\n\n            .. versionadded:: 1.6.0\n        return_length : bool, optional\n            Return the number of points inside the radius instead of a list\n            of the indices.\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        results : list or array of lists\n            If `x` is a single point, returns a list of the indices of the\n            neighbors of `x`. If `x` is an array of points, returns an object\n            array of shape tuple containing lists of neighbors.\n\n        Notes\n        -----\n        If you have many points whose neighbors you want to find, you may save\n        substantial amounts of time by putting them in a KDTree and using\n        query_ball_tree.\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> from scipy import spatial\n        >>> x, y = np.mgrid[0:5, 0:5]\n        >>> points = np.c_[x.ravel(), y.ravel()]\n        >>> tree = spatial.KDTree(points)\n        >>> sorted(tree.query_ball_point([2, 0], 1))\n        [5, 10, 11, 15]\n\n        Query multiple points and plot the results:\n\n        >>> import matplotlib.pyplot as plt\n        >>> points = np.asarray(points)\n        >>> plt.plot(points[:,0], points[:,1], '.')\n        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):\n        ...     nearby_points = points[results]\n        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')\n        >>> plt.margins(0.1, 0.1)\n        >>> plt.show()\n\n        "
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError('KDTree does not work with complex data')
        return super().query_ball_point(x, r, p, eps, workers, return_sorted, return_length)

    def query_ball_tree(self, other, r, p=2.0, eps=0):
        if False:
            while True:
                i = 10
        '\n        Find all pairs of points between `self` and `other` whose distance is\n        at most r.\n\n        Parameters\n        ----------\n        other : KDTree instance\n            The tree containing points to search against.\n        r : float\n            The maximum distance, has to be positive.\n        p : float, optional\n            Which Minkowski norm to use.  `p` has to meet the condition\n            ``1 <= p <= infinity``.\n        eps : float, optional\n            Approximate search.  Branches of the tree are not explored\n            if their nearest points are further than ``r/(1+eps)``, and\n            branches are added in bulk if their furthest points are nearer\n            than ``r * (1+eps)``.  `eps` has to be non-negative.\n\n        Returns\n        -------\n        results : list of lists\n            For each element ``self.data[i]`` of this tree, ``results[i]`` is a\n            list of the indices of its neighbors in ``other.data``.\n\n        Examples\n        --------\n        You can search all pairs of points between two kd-trees within a distance:\n\n        >>> import matplotlib.pyplot as plt\n        >>> import numpy as np\n        >>> from scipy.spatial import KDTree\n        >>> rng = np.random.default_rng()\n        >>> points1 = rng.random((15, 2))\n        >>> points2 = rng.random((15, 2))\n        >>> plt.figure(figsize=(6, 6))\n        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)\n        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)\n        >>> kd_tree1 = KDTree(points1)\n        >>> kd_tree2 = KDTree(points2)\n        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)\n        >>> for i in range(len(indexes)):\n        ...     for j in indexes[i]:\n        ...         plt.plot([points1[i, 0], points2[j, 0]],\n        ...             [points1[i, 1], points2[j, 1]], "-r")\n        >>> plt.show()\n\n        '
        return super().query_ball_tree(other, r, p, eps)

    def query_pairs(self, r, p=2.0, eps=0, output_type='set'):
        if False:
            print('Hello World!')
        'Find all pairs of points in `self` whose distance is at most r.\n\n        Parameters\n        ----------\n        r : positive float\n            The maximum distance.\n        p : float, optional\n            Which Minkowski norm to use.  `p` has to meet the condition\n            ``1 <= p <= infinity``.\n        eps : float, optional\n            Approximate search.  Branches of the tree are not explored\n            if their nearest points are further than ``r/(1+eps)``, and\n            branches are added in bulk if their furthest points are nearer\n            than ``r * (1+eps)``.  `eps` has to be non-negative.\n        output_type : string, optional\n            Choose the output container, \'set\' or \'ndarray\'. Default: \'set\'\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        results : set or ndarray\n            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding\n            positions are close. If output_type is \'ndarray\', an ndarry is\n            returned instead of a set.\n\n        Examples\n        --------\n        You can search all pairs of points in a kd-tree within a distance:\n\n        >>> import matplotlib.pyplot as plt\n        >>> import numpy as np\n        >>> from scipy.spatial import KDTree\n        >>> rng = np.random.default_rng()\n        >>> points = rng.random((20, 2))\n        >>> plt.figure(figsize=(6, 6))\n        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)\n        >>> kd_tree = KDTree(points)\n        >>> pairs = kd_tree.query_pairs(r=0.2)\n        >>> for (i, j) in pairs:\n        ...     plt.plot([points[i, 0], points[j, 0]],\n        ...             [points[i, 1], points[j, 1]], "-r")\n        >>> plt.show()\n\n        '
        return super().query_pairs(r, p, eps, output_type)

    def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
        if False:
            print('Hello World!')
        'Count how many nearby pairs can be formed.\n\n        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn\n        from ``self`` and ``x2`` drawn from ``other``, and where\n        ``distance(x1, x2, p) <= r``.\n\n        Data points on ``self`` and ``other`` are optionally weighted by the\n        ``weights`` argument. (See below)\n\n        This is adapted from the "two-point correlation" algorithm described by\n        Gray and Moore [1]_.  See notes for further discussion.\n\n        Parameters\n        ----------\n        other : KDTree\n            The other tree to draw points from, can be the same tree as self.\n        r : float or one-dimensional array of floats\n            The radius to produce a count for. Multiple radii are searched with\n            a single tree traversal.\n            If the count is non-cumulative(``cumulative=False``), ``r`` defines\n            the edges of the bins, and must be non-decreasing.\n        p : float, optional\n            1<=p<=infinity.\n            Which Minkowski p-norm to use.\n            Default 2.0.\n            A finite large p may cause a ValueError if overflow can occur.\n        weights : tuple, array_like, or None, optional\n            If None, the pair-counting is unweighted.\n            If given as a tuple, weights[0] is the weights of points in\n            ``self``, and weights[1] is the weights of points in ``other``;\n            either can be None to indicate the points are unweighted.\n            If given as an array_like, weights is the weights of points in\n            ``self`` and ``other``. For this to make sense, ``self`` and\n            ``other`` must be the same tree. If ``self`` and ``other`` are two\n            different trees, a ``ValueError`` is raised.\n            Default: None\n\n            .. versionadded:: 1.6.0\n        cumulative : bool, optional\n            Whether the returned counts are cumulative. When cumulative is set\n            to ``False`` the algorithm is optimized to work with a large number\n            of bins (>10) specified by ``r``. When ``cumulative`` is set to\n            True, the algorithm is optimized to work with a small number of\n            ``r``. Default: True\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        result : scalar or 1-D array\n            The number of pairs. For unweighted counts, the result is integer.\n            For weighted counts, the result is float.\n            If cumulative is False, ``result[i]`` contains the counts with\n            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``\n\n        Notes\n        -----\n        Pair-counting is the basic operation used to calculate the two point\n        correlation functions from a data set composed of position of objects.\n\n        Two point correlation function measures the clustering of objects and\n        is widely used in cosmology to quantify the large scale structure\n        in our Universe, but it may be useful for data analysis in other fields\n        where self-similar assembly of objects also occur.\n\n        The Landy-Szalay estimator for the two point correlation function of\n        ``D`` measures the clustering signal in ``D``. [2]_\n\n        For example, given the position of two sets of objects,\n\n        - objects ``D`` (data) contains the clustering signal, and\n\n        - objects ``R`` (random) that contains no signal,\n\n        .. math::\n\n             \\xi(r) = \\frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},\n\n        where the brackets represents counting pairs between two data sets\n        in a finite bin around ``r`` (distance), corresponding to setting\n        `cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the\n        ratio between number of objects from data and random.\n\n        The algorithm implemented here is loosely based on the dual-tree\n        algorithm described in [1]_. We switch between two different\n        pair-cumulation scheme depending on the setting of ``cumulative``.\n        The computing time of the method we use when for\n        ``cumulative == False`` does not scale with the total number of bins.\n        The algorithm for ``cumulative == True`` scales linearly with the\n        number of bins, though it is slightly faster when only\n        1 or 2 bins are used. [5]_.\n\n        As an extension to the naive pair-counting,\n        weighted pair-counting counts the product of weights instead\n        of number of pairs.\n        Weighted pair-counting is used to estimate marked correlation functions\n        ([3]_, section 2.2),\n        or to properly calculate the average of data per distance bin\n        (e.g. [4]_, section 2.1 on redshift).\n\n        .. [1] Gray and Moore,\n               "N-body problems in statistical learning",\n               Mining the sky, 2000,\n               https://arxiv.org/abs/astro-ph/0012333\n\n        .. [2] Landy and Szalay,\n               "Bias and variance of angular correlation functions",\n               The Astrophysical Journal, 1993,\n               http://adsabs.harvard.edu/abs/1993ApJ...412...64L\n\n        .. [3] Sheth, Connolly and Skibba,\n               "Marked correlations in galaxy formation models",\n               Arxiv e-print, 2005,\n               https://arxiv.org/abs/astro-ph/0511773\n\n        .. [4] Hawkins, et al.,\n               "The 2dF Galaxy Redshift Survey: correlation functions,\n               peculiar velocities and the matter density of the Universe",\n               Monthly Notices of the Royal Astronomical Society, 2002,\n               http://adsabs.harvard.edu/abs/2003MNRAS.346...78H\n\n        .. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926\n\n        Examples\n        --------\n        You can count neighbors number between two kd-trees within a distance:\n\n        >>> import numpy as np\n        >>> from scipy.spatial import KDTree\n        >>> rng = np.random.default_rng()\n        >>> points1 = rng.random((5, 2))\n        >>> points2 = rng.random((5, 2))\n        >>> kd_tree1 = KDTree(points1)\n        >>> kd_tree2 = KDTree(points2)\n        >>> kd_tree1.count_neighbors(kd_tree2, 0.2)\n        1\n\n        This number is same as the total pair number calculated by\n        `query_ball_tree`:\n\n        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)\n        >>> sum([len(i) for i in indexes])\n        1\n\n        '
        return super().count_neighbors(other, r, p, weights, cumulative)

    def sparse_distance_matrix(self, other, max_distance, p=2.0, output_type='dok_matrix'):
        if False:
            for i in range(10):
                print('nop')
        'Compute a sparse distance matrix.\n\n        Computes a distance matrix between two KDTrees, leaving as zero\n        any distance greater than max_distance.\n\n        Parameters\n        ----------\n        other : KDTree\n\n        max_distance : positive float\n\n        p : float, 1<=p<=infinity\n            Which Minkowski p-norm to use.\n            A finite large p may cause a ValueError if overflow can occur.\n\n        output_type : string, optional\n            Which container to use for output data. Options: \'dok_matrix\',\n            \'coo_matrix\', \'dict\', or \'ndarray\'. Default: \'dok_matrix\'.\n\n            .. versionadded:: 1.6.0\n\n        Returns\n        -------\n        result : dok_matrix, coo_matrix, dict or ndarray\n            Sparse matrix representing the results in "dictionary of keys"\n            format. If a dict is returned the keys are (i,j) tuples of indices.\n            If output_type is \'ndarray\' a record array with fields \'i\', \'j\',\n            and \'v\' is returned,\n\n        Examples\n        --------\n        You can compute a sparse distance matrix between two kd-trees:\n\n        >>> import numpy as np\n        >>> from scipy.spatial import KDTree\n        >>> rng = np.random.default_rng()\n        >>> points1 = rng.random((5, 2))\n        >>> points2 = rng.random((5, 2))\n        >>> kd_tree1 = KDTree(points1)\n        >>> kd_tree2 = KDTree(points2)\n        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)\n        >>> sdm.toarray()\n        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],\n           [0.        , 0.        , 0.        , 0.        , 0.        ],\n           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],\n           [0.        , 0.        , 0.        , 0.        , 0.        ],\n           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])\n\n        You can check distances above the `max_distance` are zeros:\n\n        >>> from scipy.spatial import distance_matrix\n        >>> distance_matrix(points1, points2)\n        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],\n           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],\n           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],\n           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],\n           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])\n\n        '
        return super().sparse_distance_matrix(other, max_distance, p, output_type)

def distance_matrix(x, y, p=2, threshold=1000000):
    if False:
        for i in range(10):
            print('nop')
    'Compute the distance matrix.\n\n    Returns the matrix of all pair-wise distances.\n\n    Parameters\n    ----------\n    x : (M, K) array_like\n        Matrix of M vectors in K dimensions.\n    y : (N, K) array_like\n        Matrix of N vectors in K dimensions.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n    threshold : positive int\n        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead\n        of large temporary arrays.\n\n    Returns\n    -------\n    result : (M, N) ndarray\n        Matrix containing the distance from every vector in `x` to every vector\n        in `y`.\n\n    Examples\n    --------\n    >>> from scipy.spatial import distance_matrix\n    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])\n    array([[ 1.        ,  1.41421356],\n           [ 1.41421356,  1.        ]])\n\n    '
    x = np.asarray(x)
    (m, k) = x.shape
    y = np.asarray(y)
    (n, kk) = y.shape
    if k != kk:
        raise ValueError(f'x contains {k}-dimensional vectors but y contains {kk}-dimensional vectors')
    if m * n * k <= threshold:
        return minkowski_distance(x[:, np.newaxis, :], y[np.newaxis, :, :], p)
    else:
        result = np.empty((m, n), dtype=float)
        if m < n:
            for i in range(m):
                result[i, :] = minkowski_distance(x[i], y, p)
        else:
            for j in range(n):
                result[:, j] = minkowski_distance(x, y[j], p)
        return result