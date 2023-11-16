import collections

class Solution:

    def topologicalSortingDFS(self, graph: dict):
        if False:
            for i in range(10):
                print('nop')
        visited = set()
        onStack = set()
        order = []
        hasCycle = False

        def dfs(u):
            if False:
                print('Hello World!')
            nonlocal hasCycle
            if u in onStack:
                hasCycle = True
            if u in visited or hasCycle:
                return
            visited.add(u)
            onStack.add(u)
            for v in graph[u]:
                dfs(v)
            order.append(u)
            onStack.remove(u)
        for u in graph:
            if u not in visited:
                dfs(u)
        if hasCycle:
            return []
        order.reverse()
        return order

    def findOrder(self, n: int, edges):
        if False:
            return 10
        graph = dict()
        for i in range(n):
            graph[i] = []
        for (v, u) in edges:
            graph[u].append(v)
        return self.topologicalSortingDFS(graph)
print(Solution().findOrder(2, [[1, 0]]))
print(Solution().findOrder(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
print(Solution().findOrder(1, []))