from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict

class Prufer(Basic):
    """
    The Prufer correspondence is an algorithm that describes the
    bijection between labeled trees and the Prufer code. A Prufer
    code of a labeled tree is unique up to isomorphism and has
    a length of n - 2.

    Prufer sequences were first used by Heinz Prufer to give a
    proof of Cayley's formula.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/LabeledTree.html

    """
    _prufer_repr = None
    _tree_repr = None
    _nodes = None
    _rank = None

    @property
    def prufer_repr(self):
        if False:
            print('Hello World!')
        'Returns Prufer sequence for the Prufer object.\n\n        This sequence is found by removing the highest numbered vertex,\n        recording the node it was attached to, and continuing until only\n        two vertices remain. The Prufer sequence is the list of recorded nodes.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).prufer_repr\n        [3, 3, 3, 4]\n        >>> Prufer([1, 0, 0]).prufer_repr\n        [1, 0, 0]\n\n        See Also\n        ========\n\n        to_prufer\n\n        '
        if self._prufer_repr is None:
            self._prufer_repr = self.to_prufer(self._tree_repr[:], self.nodes)
        return self._prufer_repr

    @property
    def tree_repr(self):
        if False:
            return 10
        'Returns the tree representation of the Prufer object.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).tree_repr\n        [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]\n        >>> Prufer([1, 0, 0]).tree_repr\n        [[1, 2], [0, 1], [0, 3], [0, 4]]\n\n        See Also\n        ========\n\n        to_tree\n\n        '
        if self._tree_repr is None:
            self._tree_repr = self.to_tree(self._prufer_repr[:])
        return self._tree_repr

    @property
    def nodes(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of nodes in the tree.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).nodes\n        6\n        >>> Prufer([1, 0, 0]).nodes\n        5\n\n        '
        return self._nodes

    @property
    def rank(self):
        if False:
            i = 10
            return i + 15
        'Returns the rank of the Prufer sequence.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> p = Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]])\n        >>> p.rank\n        778\n        >>> p.next(1).rank\n        779\n        >>> p.prev().rank\n        777\n\n        See Also\n        ========\n\n        prufer_rank, next, prev, size\n\n        '
        if self._rank is None:
            self._rank = self.prufer_rank()
        return self._rank

    @property
    def size(self):
        if False:
            print('Hello World!')
        'Return the number of possible trees of this Prufer object.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer([0]*4).size == Prufer([6]*4).size == 1296\n        True\n\n        See Also\n        ========\n\n        prufer_rank, rank, next, prev\n\n        '
        return self.prev(self.rank).prev().rank + 1

    @staticmethod
    def to_prufer(tree, n):
        if False:
            i = 10
            return i + 15
        'Return the Prufer sequence for a tree given as a list of edges where\n        ``n`` is the number of nodes in the tree.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])\n        >>> a.prufer_repr\n        [0, 0]\n        >>> Prufer.to_prufer([[0, 1], [0, 2], [0, 3]], 4)\n        [0, 0]\n\n        See Also\n        ========\n        prufer_repr: returns Prufer sequence of a Prufer object.\n\n        '
        d = defaultdict(int)
        L = []
        for edge in tree:
            d[edge[0]] += 1
            d[edge[1]] += 1
        for i in range(n - 2):
            for x in range(n):
                if d[x] == 1:
                    break
            y = None
            for edge in tree:
                if x == edge[0]:
                    y = edge[1]
                elif x == edge[1]:
                    y = edge[0]
                if y is not None:
                    break
            L.append(y)
            for j in (x, y):
                d[j] -= 1
                if not d[j]:
                    d.pop(j)
            tree.remove(edge)
        return L

    @staticmethod
    def to_tree(prufer):
        if False:
            print('Hello World!')
        'Return the tree (as a list of edges) of the given Prufer sequence.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> a = Prufer([0, 2], 4)\n        >>> a.tree_repr\n        [[0, 1], [0, 2], [2, 3]]\n        >>> Prufer.to_tree([0, 2])\n        [[0, 1], [0, 2], [2, 3]]\n\n        References\n        ==========\n\n        .. [1] https://hamberg.no/erlend/posts/2010-11-06-prufer-sequence-compact-tree-representation.html\n\n        See Also\n        ========\n        tree_repr: returns tree representation of a Prufer object.\n\n        '
        tree = []
        last = []
        n = len(prufer) + 2
        d = defaultdict(lambda : 1)
        for p in prufer:
            d[p] += 1
        for i in prufer:
            for j in range(n):
                if d[j] == 1:
                    break
            d[i] -= 1
            d[j] -= 1
            tree.append(sorted([i, j]))
        last = [i for i in range(n) if d[i] == 1] or [0, 1]
        tree.append(last)
        return tree

    @staticmethod
    def edges(*runs):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of edges and the number of nodes from the given runs\n        that connect nodes in an integer-labelled tree.\n\n        All node numbers will be shifted so that the minimum node is 0. It is\n        not a problem if edges are repeated in the runs; only unique edges are\n        returned. There is no assumption made about what the range of the node\n        labels should be, but all nodes from the smallest through the largest\n        must be present.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer.edges([1, 2, 3], [2, 4, 5]) # a T\n        ([[0, 1], [1, 2], [1, 3], [3, 4]], 5)\n\n        Duplicate edges are removed:\n\n        >>> Prufer.edges([0, 1, 2, 3], [1, 4, 5], [1, 4, 6]) # a K\n        ([[0, 1], [1, 2], [1, 4], [2, 3], [4, 5], [4, 6]], 7)\n\n        '
        e = set()
        nmin = runs[0][0]
        for r in runs:
            for i in range(len(r) - 1):
                (a, b) = r[i:i + 2]
                if b < a:
                    (a, b) = (b, a)
                e.add((a, b))
        rv = []
        got = set()
        nmin = nmax = None
        for ei in e:
            for i in ei:
                got.add(i)
            nmin = min(ei[0], nmin) if nmin is not None else ei[0]
            nmax = max(ei[1], nmax) if nmax is not None else ei[1]
            rv.append(list(ei))
        missing = set(range(nmin, nmax + 1)) - got
        if missing:
            missing = [i + nmin for i in missing]
            if len(missing) == 1:
                msg = 'Node %s is missing.' % missing.pop()
            else:
                msg = 'Nodes %s are missing.' % sorted(missing)
            raise ValueError(msg)
        if nmin != 0:
            for (i, ei) in enumerate(rv):
                rv[i] = [n - nmin for n in ei]
            nmax -= nmin
        return (sorted(rv), nmax + 1)

    def prufer_rank(self):
        if False:
            return 10
        'Computes the rank of a Prufer sequence.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])\n        >>> a.prufer_rank()\n        0\n\n        See Also\n        ========\n\n        rank, next, prev, size\n\n        '
        r = 0
        p = 1
        for i in range(self.nodes - 3, -1, -1):
            r += p * self.prufer_repr[i]
            p *= self.nodes
        return r

    @classmethod
    def unrank(self, rank, n):
        if False:
            while True:
                i = 10
        'Finds the unranked Prufer sequence.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> Prufer.unrank(0, 4)\n        Prufer([0, 0])\n\n        '
        (n, rank) = (as_int(n), as_int(rank))
        L = defaultdict(int)
        for i in range(n - 3, -1, -1):
            L[i] = rank % n
            rank = (rank - L[i]) // n
        return Prufer([L[i] for i in range(len(L))])

    def __new__(cls, *args, **kw_args):
        if False:
            print('Hello World!')
        'The constructor for the Prufer object.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n\n        A Prufer object can be constructed from a list of edges:\n\n        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])\n        >>> a.prufer_repr\n        [0, 0]\n\n        If the number of nodes is given, no checking of the nodes will\n        be performed; it will be assumed that nodes 0 through n - 1 are\n        present:\n\n        >>> Prufer([[0, 1], [0, 2], [0, 3]], 4)\n        Prufer([[0, 1], [0, 2], [0, 3]], 4)\n\n        A Prufer object can be constructed from a Prufer sequence:\n\n        >>> b = Prufer([1, 3])\n        >>> b.tree_repr\n        [[0, 1], [1, 3], [2, 3]]\n\n        '
        arg0 = Array(args[0]) if args[0] else Tuple()
        args = (arg0,) + tuple((_sympify(arg) for arg in args[1:]))
        ret_obj = Basic.__new__(cls, *args, **kw_args)
        args = [list(args[0])]
        if args[0] and iterable(args[0][0]):
            if not args[0][0]:
                raise ValueError('Prufer expects at least one edge in the tree.')
            if len(args) > 1:
                nnodes = args[1]
            else:
                nodes = set(flatten(args[0]))
                nnodes = max(nodes) + 1
                if nnodes != len(nodes):
                    missing = set(range(nnodes)) - nodes
                    if len(missing) == 1:
                        msg = 'Node %s is missing.' % missing.pop()
                    else:
                        msg = 'Nodes %s are missing.' % sorted(missing)
                    raise ValueError(msg)
            ret_obj._tree_repr = [list(i) for i in args[0]]
            ret_obj._nodes = nnodes
        else:
            ret_obj._prufer_repr = args[0]
            ret_obj._nodes = len(ret_obj._prufer_repr) + 2
        return ret_obj

    def next(self, delta=1):
        if False:
            while True:
                i = 10
        'Generates the Prufer sequence that is delta beyond the current one.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])\n        >>> b = a.next(1) # == a.next()\n        >>> b.tree_repr\n        [[0, 2], [0, 1], [1, 3]]\n        >>> b.rank\n        1\n\n        See Also\n        ========\n\n        prufer_rank, rank, prev, size\n\n        '
        return Prufer.unrank(self.rank + delta, self.nodes)

    def prev(self, delta=1):
        if False:
            for i in range(10):
                print('nop')
        'Generates the Prufer sequence that is -delta before the current one.\n\n        Examples\n        ========\n\n        >>> from sympy.combinatorics.prufer import Prufer\n        >>> a = Prufer([[0, 1], [1, 2], [2, 3], [1, 4]])\n        >>> a.rank\n        36\n        >>> b = a.prev()\n        >>> b\n        Prufer([1, 2, 0])\n        >>> b.rank\n        35\n\n        See Also\n        ========\n\n        prufer_rank, rank, next, size\n\n        '
        return Prufer.unrank(self.rank - delta, self.nodes)