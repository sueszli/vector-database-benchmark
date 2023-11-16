import collections

class Solution(object):

    def sumOfDistancesInTree(self, N, edges):
        if False:
            return 10
        '\n        :type N: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def dfs(graph, node, parent, count, result):
            if False:
                while True:
                    i = 10
            for nei in graph[node]:
                if nei != parent:
                    dfs(graph, nei, node, count, result)
                    count[node] += count[nei]
                    result[node] += result[nei] + count[nei]

        def dfs2(graph, node, parent, count, result):
            if False:
                while True:
                    i = 10
            for nei in graph[node]:
                if nei != parent:
                    result[nei] = result[node] - count[nei] + len(count) - count[nei]
                    dfs2(graph, nei, node, count, result)
        graph = collections.defaultdict(list)
        for (u, v) in edges:
            graph[u].append(v)
            graph[v].append(u)
        count = [1] * N
        result = [0] * N
        dfs(graph, 0, None, count, result)
        dfs2(graph, 0, None, count, result)
        return result