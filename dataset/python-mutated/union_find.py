"""
Union-find data structure.
"""
from networkx.utils import groups

class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      https://code.activestate.com/recipes/215912/
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self, elements=None):
        if False:
            print('Hello World!')
        'Create a new empty union-find structure.\n\n        If *elements* is an iterable, this structure will be initialized\n        with the discrete partition on the given set of elements.\n\n        '
        if elements is None:
            elements = ()
        self.parents = {}
        self.weights = {}
        for x in elements:
            self.weights[x] = 1
            self.parents[x] = x

    def __getitem__(self, object):
        if False:
            print('Hello World!')
        'Find and return the name of the set containing the object.'
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object
        path = []
        root = self.parents[object]
        while root != object:
            path.append(object)
            object = root
            root = self.parents[object]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate through all items ever found or unioned by this structure.'
        return iter(self.parents)

    def to_sets(self):
        if False:
            i = 10
            return i + 15
        'Iterates over the sets stored in this structure.\n\n        For example::\n\n            >>> partition = UnionFind("xyz")\n            >>> sorted(map(sorted, partition.to_sets()))\n            [[\'x\'], [\'y\'], [\'z\']]\n            >>> partition.union("x", "y")\n            >>> sorted(map(sorted, partition.to_sets()))\n            [[\'x\', \'y\'], [\'z\']]\n\n        '
        for x in self.parents:
            _ = self[x]
        yield from groups(self.parents).values()

    def union(self, *objects):
        if False:
            print('Hello World!')
        'Find the sets containing the objects and merge them all.'
        roots = iter(sorted({self[x] for x in objects}, key=lambda r: self.weights[r], reverse=True))
        try:
            root = next(roots)
        except StopIteration:
            return
        for r in roots:
            self.weights[root] += self.weights[r]
            self.parents[r] = root