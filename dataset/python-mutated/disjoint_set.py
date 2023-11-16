"""
    In computer science, a disjoint-set data structure, also called a union–find data structure or merge–find set,
    is a data structure that stores a collection of disjoint (non-overlapping) sets.
"""
import typing

class disjointset:
    """
    In computer science, a disjoint-set data structure, also called a union–find data structure or merge–find set,
    is a data structure that stores a collection of disjoint (non-overlapping) sets.
    Equivalently, it stores a partition of a set into disjoint subsets.
    It provides operations for adding new sets, merging sets (replacing them by their union),
    and finding a representative member of a set.
    The last operation allows to find out efficiently if any two elements are in the same or different sets.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._parents = {}
        self._ranks = {}

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return item in self._parents

    def __iter__(self):
        if False:
            print('Hello World!')
        return self._parents.__iter__()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._parents)

    def add(self, x: typing.Any) -> 'disjointset':
        if False:
            return 10
        '\n        Add an element to this disjointset\n        '
        self._parents[x] = x
        self._ranks[x] = 0
        return self

    def find(self, x: typing.Any) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        Find the root of an element in this disjointset\n        '
        if self._parents[x] == x:
            return x
        else:
            return self.find(self._parents[x])

    def pop(self, x: typing.Any) -> 'disjointset':
        if False:
            i = 10
            return i + 15
        '\n        Remove an element from this disjointset\n        '
        raise NotImplementedError()

    def sets(self) -> typing.List[typing.List[typing.Any]]:
        if False:
            return 10
        '\n        This function returns all equivalence sets in this disjointset\n        '
        cluster_parents: typing.Dict[typing.Any, typing.Any] = {}
        for (x, _) in self._parents.items():
            p = self.find(x)
            if p not in cluster_parents:
                cluster_parents[p] = []
            cluster_parents[p].append(x)
        return [v for (k, v) in cluster_parents.items()]

    def union(self, x: typing.Any, y: typing.Any) -> 'disjointset':
        if False:
            print('Hello World!')
        '\n        Mark two elements in this disjointset as equivalent,\n        propagating the equivalence throughout the disjointset\n        '
        x_parent = self.find(x)
        y_parent = self.find(y)
        if x_parent is y_parent:
            return self
        if self._ranks[x_parent] > self._ranks[y_parent]:
            self._parents[y_parent] = x_parent
        elif self._ranks[y_parent] > self._ranks[x_parent]:
            self._parents[x_parent] = y_parent
        else:
            self._parents[y_parent] = x_parent
            self._ranks[x_parent] += 1
        return self