import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation

class TriFinder:
    """
    Abstract base class for classes used to find the triangles of a
    Triangulation in which (x, y) points lie.

    Rather than instantiate an object of a class derived from TriFinder, it is
    usually better to use the function `.Triangulation.get_trifinder`.

    Derived classes implement __call__(x, y) where x and y are array-like point
    coordinates of the same shape.
    """

    def __init__(self, triangulation):
        if False:
            print('Hello World!')
        _api.check_isinstance(Triangulation, triangulation=triangulation)
        self._triangulation = triangulation

    def __call__(self, x, y):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class TrapezoidMapTriFinder(TriFinder):
    """
    `~matplotlib.tri.TriFinder` class implemented using the trapezoid
    map algorithm from the book "Computational Geometry, Algorithms and
    Applications", second edition, by M. de Berg, M. van Kreveld, M. Overmars
    and O. Schwarzkopf.

    The triangulation must be valid, i.e. it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.  The
    algorithm has some tolerance to triangles formed from colinear points, but
    this should not be relied upon.
    """

    def __init__(self, triangulation):
        if False:
            print('Hello World!')
        from matplotlib import _tri
        super().__init__(triangulation)
        self._cpp_trifinder = _tri.TrapezoidMapTriFinder(triangulation.get_cpp_triangulation())
        self._initialize()

    def __call__(self, x, y):
        if False:
            i = 10
            return i + 15
        '\n        Return an array containing the indices of the triangles in which the\n        specified *x*, *y* points lie, or -1 for points that do not lie within\n        a triangle.\n\n        *x*, *y* are array-like x and y coordinates of the same shape and any\n        number of dimensions.\n\n        Returns integer array with the same shape and *x* and *y*.\n        '
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError('x and y must be array-like with the same shape')
        indices = self._cpp_trifinder.find_many(x.ravel(), y.ravel()).reshape(x.shape)
        return indices

    def _get_tree_stats(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a python list containing the statistics about the node tree:\n            0: number of nodes (tree size)\n            1: number of unique nodes\n            2: number of trapezoids (tree leaf nodes)\n            3: number of unique trapezoids\n            4: maximum parent count (max number of times a node is repeated in\n                   tree)\n            5: maximum depth of tree (one more than the maximum number of\n                   comparisons needed to search through the tree)\n            6: mean of all trapezoid depths (one more than the average number\n                   of comparisons needed to search through the tree)\n        '
        return self._cpp_trifinder.get_tree_stats()

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the underlying C++ object.  Can be called multiple times if,\n        for example, the triangulation is modified.\n        '
        self._cpp_trifinder.initialize()

    def _print_tree(self):
        if False:
            print('Hello World!')
        '\n        Print a text representation of the node tree, which is useful for\n        debugging purposes.\n        '
        self._cpp_trifinder.print_tree()