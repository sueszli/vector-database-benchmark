"""
Disjoint set data structure
"""

class DisjointSet:
    """ Disjoint set data structure for incremental connectivity queries.

    .. versionadded:: 1.6.0

    Attributes
    ----------
    n_subsets : int
        The number of subsets.

    Methods
    -------
    add
    merge
    connected
    subset
    subset_size
    subsets
    __getitem__

    Notes
    -----
    This class implements the disjoint set [1]_, also known as the *union-find*
    or *merge-find* data structure. The *find* operation (implemented in
    `__getitem__`) implements the *path halving* variant. The *merge* method
    implements the *merge by size* variant.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Examples
    --------
    >>> from scipy.cluster.hierarchy import DisjointSet

    Initialize a disjoint set:

    >>> disjoint_set = DisjointSet([1, 2, 3, 'a', 'b'])

    Merge some subsets:

    >>> disjoint_set.merge(1, 2)
    True
    >>> disjoint_set.merge(3, 'a')
    True
    >>> disjoint_set.merge('a', 'b')
    True
    >>> disjoint_set.merge('b', 'b')
    False

    Find root elements:

    >>> disjoint_set[2]
    1
    >>> disjoint_set['b']
    3

    Test connectivity:

    >>> disjoint_set.connected(1, 2)
    True
    >>> disjoint_set.connected(1, 'b')
    False

    List elements in disjoint set:

    >>> list(disjoint_set)
    [1, 2, 3, 'a', 'b']

    Get the subset containing 'a':

    >>> disjoint_set.subset('a')
    {'a', 3, 'b'}

    Get the size of the subset containing 'a' (without actually instantiating
    the subset):

    >>> disjoint_set.subset_size('a')
    3

    Get all subsets in the disjoint set:

    >>> disjoint_set.subsets()
    [{1, 2}, {'a', 3, 'b'}]
    """

    def __init__(self, elements=None):
        if False:
            return 10
        self.n_subsets = 0
        self._sizes = {}
        self._parents = {}
        self._nbrs = {}
        self._indices = {}
        if elements is not None:
            for x in elements:
                self.add(x)

    def __iter__(self):
        if False:
            return 10
        'Returns an iterator of the elements in the disjoint set.\n\n        Elements are ordered by insertion order.\n        '
        return iter(self._indices)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._indices)

    def __contains__(self, x):
        if False:
            print('Hello World!')
        return x in self._indices

    def __getitem__(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Find the root element of `x`.\n\n        Parameters\n        ----------\n        x : hashable object\n            Input element.\n\n        Returns\n        -------\n        root : hashable object\n            Root element of `x`.\n        '
        if x not in self._indices:
            raise KeyError(x)
        parents = self._parents
        while self._indices[x] != self._indices[parents[x]]:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def add(self, x):
        if False:
            while True:
                i = 10
        'Add element `x` to disjoint set\n        '
        if x in self._indices:
            return
        self._sizes[x] = 1
        self._parents[x] = x
        self._nbrs[x] = x
        self._indices[x] = len(self._indices)
        self.n_subsets += 1

    def merge(self, x, y):
        if False:
            print('Hello World!')
        'Merge the subsets of `x` and `y`.\n\n        The smaller subset (the child) is merged into the larger subset (the\n        parent). If the subsets are of equal size, the root element which was\n        first inserted into the disjoint set is selected as the parent.\n\n        Parameters\n        ----------\n        x, y : hashable object\n            Elements to merge.\n\n        Returns\n        -------\n        merged : bool\n            True if `x` and `y` were in disjoint sets, False otherwise.\n        '
        xr = self[x]
        yr = self[y]
        if self._indices[xr] == self._indices[yr]:
            return False
        sizes = self._sizes
        if (sizes[xr], self._indices[yr]) < (sizes[yr], self._indices[xr]):
            (xr, yr) = (yr, xr)
        self._parents[yr] = xr
        self._sizes[xr] += self._sizes[yr]
        (self._nbrs[xr], self._nbrs[yr]) = (self._nbrs[yr], self._nbrs[xr])
        self.n_subsets -= 1
        return True

    def connected(self, x, y):
        if False:
            return 10
        'Test whether `x` and `y` are in the same subset.\n\n        Parameters\n        ----------\n        x, y : hashable object\n            Elements to test.\n\n        Returns\n        -------\n        result : bool\n            True if `x` and `y` are in the same set, False otherwise.\n        '
        return self._indices[self[x]] == self._indices[self[y]]

    def subset(self, x):
        if False:
            return 10
        'Get the subset containing `x`.\n\n        Parameters\n        ----------\n        x : hashable object\n            Input element.\n\n        Returns\n        -------\n        result : set\n            Subset containing `x`.\n        '
        if x not in self._indices:
            raise KeyError(x)
        result = [x]
        nxt = self._nbrs[x]
        while self._indices[nxt] != self._indices[x]:
            result.append(nxt)
            nxt = self._nbrs[nxt]
        return set(result)

    def subset_size(self, x):
        if False:
            i = 10
            return i + 15
        'Get the size of the subset containing `x`.\n\n        Note that this method is faster than ``len(self.subset(x))`` because\n        the size is directly read off an internal field, without the need to\n        instantiate the full subset.\n\n        Parameters\n        ----------\n        x : hashable object\n            Input element.\n\n        Returns\n        -------\n        result : int\n            Size of the subset containing `x`.\n        '
        return self._sizes[self[x]]

    def subsets(self):
        if False:
            for i in range(10):
                print('nop')
        'Get all the subsets in the disjoint set.\n\n        Returns\n        -------\n        result : list\n            Subsets in the disjoint set.\n        '
        result = []
        visited = set()
        for x in self:
            if x not in visited:
                xset = self.subset(x)
                visited.update(xset)
                result.append(xset)
        return result