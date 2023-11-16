"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""

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
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        'Create a new empty union-find structure.'
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if False:
            print('Hello World!')
        'Find and return the name of the set containing the object.'
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Iterate through all items ever found or unioned by this structure.'
        return iter(self.parents)

    def union(self, *objects):
        if False:
            for i in range(10):
                print('nop')
        'Find the sets containing the objects and merge them all.'
        roots = [self[x] for x in objects]
        heaviest = max(((self.weights[r], r) for r in roots))[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest