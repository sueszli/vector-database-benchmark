import itertools

class Solution(object):

    def minNumberOfSemesters(self, n, dependencies, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type dependencies: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        reqs = [0] * n
        for (u, v) in dependencies:
            reqs[v - 1] |= 1 << u - 1
        dp = [n] * (1 << n)
        dp[0] = 0
        for mask in xrange(1 << n):
            candidates = []
            for v in xrange(n):
                if mask & 1 << v == 0 and mask & reqs[v] == reqs[v]:
                    candidates.append(v)
            for choice in itertools.combinations(candidates, min(len(candidates), k)):
                new_mask = mask
                for v in choice:
                    new_mask |= 1 << v
                dp[new_mask] = min(dp[new_mask], dp[mask] + 1)
        return dp[-1]
import collections
import heapq

class Solution_WA(object):

    def minNumberOfSemesters(self, n, dependencies, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type dependencies: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        def dfs(graph, i, depths):
            if False:
                for i in range(10):
                    print('nop')
            if depths[i] == -1:
                depths[i] = max((dfs(graph, child, depths) for child in graph[i])) + 1 if i in graph else 1
            return depths[i]
        degrees = [0] * n
        graph = collections.defaultdict(list)
        for (u, v) in dependencies:
            graph[u - 1].append(v - 1)
            degrees[v - 1] += 1
        depths = [-1] * n
        for i in xrange(n):
            dfs(graph, i, depths)
        max_heap = []
        for i in xrange(n):
            if not degrees[i]:
                heapq.heappush(max_heap, (-depths[i], i))
        result = 0
        while max_heap:
            new_q = []
            for _ in xrange(min(len(max_heap), k)):
                (_, node) = heapq.heappop(max_heap)
                if node not in graph:
                    continue
                for child in graph[node]:
                    degrees[child] -= 1
                    if not degrees[child]:
                        new_q.append(child)
            result += 1
            for node in new_q:
                heapq.heappush(max_heap, (-depths[node], node))
        return result