import collections

class Solution:

    def topologicalSortingKahn(self, graph: dict):
        if False:
            while True:
                i = 10
        indegrees = {u: 0 for u in graph}
        for u in graph:
            for v in graph[u]:
                indegrees[v] += 1
        S = collections.deque([u for u in indegrees if indegrees[u] == 0])
        order = []
        while S:
            u = S.pop()
            order.append(u)
            for v in graph[u]:
                indegrees[v] -= 1
                if indegrees[v] == 0:
                    S.append(v)
        if len(indegrees) != len(order):
            return []
        return order

    def findOrder(self, n: int, edges):
        if False:
            i = 10
            return i + 15
        graph = dict()
        for i in range(n):
            graph[i] = []
        for (u, v) in edges:
            graph[u].append(v)
        return self.topologicalSortingKahn(graph)
print(Solution().findOrder(2, [[1, 0]]))
print(Solution().findOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
print(Solution().findOrder(1, []))