class UnionFind:
    """Union-find data structure.

    Union-find is a data structure that keeps track of a set of elements partitioned
    into a number of disjoint (non-overlapping) subsets.

    Reference:
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Args:
      elements(list): The initialize element list.
    """

    def __init__(self, elementes=None):
        if False:
            while True:
                i = 10
        self._parents = []
        self._index = {}
        self._curr_idx = 0
        if not elementes:
            elementes = []
        for ele in elementes:
            self._parents.append(self._curr_idx)
            self._index.update({ele: self._curr_idx})
            self._curr_idx += 1

    def find(self, x):
        if False:
            for i in range(10):
                print('nop')
        if x not in self._index:
            return -1
        idx = self._index[x]
        while idx != self._parents[idx]:
            t = self._parents[idx]
            self._parents[idx] = self._parents[t]
            idx = t
        return idx

    def union(self, x, y):
        if False:
            print('Hello World!')
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        self._parents[x_root] = y_root

    def is_connected(self, x, y):
        if False:
            i = 10
            return i + 15
        return self.find(x) == self.find(y)