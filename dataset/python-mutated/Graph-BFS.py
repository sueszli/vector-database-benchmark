import collections

class Solution:

    def bfs(self, graph, u):
        if False:
            while True:
                i = 10
        visited = set()
        queue = collections.deque([])
        visited.add(u)
        queue.append(u)
        while queue:
            u = queue.popleft()
            print(u)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
graph = {'0': ['1', '2'], '1': ['0', '2', '3'], '2': ['0', '1', '3', '4'], '3': ['1', '2', '4', '5'], '4': ['2', '3'], '5': ['3', '6'], '6': []}
Solution().bfs(graph, '0')