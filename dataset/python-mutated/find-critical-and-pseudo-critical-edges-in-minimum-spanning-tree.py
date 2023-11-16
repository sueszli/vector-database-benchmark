class UnionFind(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.set = range(n)
        self.count = n

    def find_set(self, x):
        if False:
            print('Hello World!')
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            i = 10
            return i + 15
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        self.count -= 1
        return True

class Solution(object):

    def findCriticalAndPseudoCriticalEdges(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def MST(n, edges, unused=None, used=None):
            if False:
                for i in range(10):
                    print('nop')
            union_find = UnionFind(n)
            weight = 0
            if used is not None:
                (u, v, w, _) = edges[used]
                if union_find.union_set(u, v):
                    weight += w
            for (i, (u, v, w, _)) in enumerate(edges):
                if i == unused:
                    continue
                if union_find.union_set(u, v):
                    weight += w
            return weight if union_find.count == 1 else float('inf')
        for (i, edge) in enumerate(edges):
            edge.append(i)
        edges.sort(key=lambda x: x[2])
        mst = MST(n, edges)
        result = [[], []]
        for (i, edge) in enumerate(edges):
            if mst < MST(n, edges, unused=i):
                result[0].append(edge[3])
            elif mst == MST(n, edges, used=i):
                result[1].append(edge[3])
        return result