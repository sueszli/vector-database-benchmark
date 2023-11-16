class UnionFind(object):

    def __init__(self, n):
        if False:
            return 10
        self.set = range(n)

    def find_set(self, x):
        if False:
            return 10
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            print('Hello World!')
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        return True

class Solution(object):

    def findRedundantDirectedConnection(self, edges):
        if False:
            print('Hello World!')
        '\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '
        (cand1, cand2) = ([], [])
        parent = {}
        for edge in edges:
            if edge[1] not in parent:
                parent[edge[1]] = edge[0]
            else:
                cand1 = [parent[edge[1]], edge[1]]
                cand2 = edge
        union_find = UnionFind(len(edges) + 1)
        for edge in edges:
            if edge == cand2:
                continue
            if not union_find.union_set(*edge):
                return cand1 if cand2 else edge
        return cand2