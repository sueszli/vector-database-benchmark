class Solution:

    def dfs_recursive(self, graph, u, visited):
        if False:
            i = 10
            return i + 15
        print(u)
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                self.dfs_recursive(graph, v, visited)

    def dfs_stack(self, graph, u):
        if False:
            return 10
        print(u)
        (visited, stack) = (set(), [])
        stack.append([u, 0])
        visited.add(u)
        while stack:
            (u, i) = stack.pop()
            if i < len(graph[u]):
                v = graph[u][i]
                stack.append([u, i + 1])
                if v not in visited:
                    print(v)
                    stack.append([v, 0])
                    visited.add(v)
graph = {'A': ['B', 'C'], 'B': ['A', 'C', 'D'], 'C': ['A', 'B', 'D', 'E'], 'D': ['B', 'C', 'E', 'F'], 'E': ['C', 'D'], 'F': ['D', 'G'], 'G': []}
visited = set()
Solution().dfs_recursive(graph, 'A', visited)
Solution().dfs_stack(graph, 'A')