import collections

class Solution(object):

    def minReorder(self, n, connections):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type connections: List[List[int]]\n        :rtype: int\n        '
        (lookup, graph) = (set(), collections.defaultdict(list))
        for (u, v) in connections:
            lookup.add(u * n + v)
            graph[v].append(u)
            graph[u].append(v)
        result = 0
        stk = [(-1, 0)]
        while stk:
            (parent, u) = stk.pop()
            result += parent * n + u in lookup
            for v in reversed(graph[u]):
                if v == parent:
                    continue
                stk.append((u, v))
        return result
import collections

class Solution2(object):

    def minReorder(self, n, connections):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type connections: List[List[int]]\n        :rtype: int\n        '

        def dfs(n, lookup, graph, parent, u):
            if False:
                for i in range(10):
                    print('nop')
            result = parent * n + u in lookup
            for v in graph[u]:
                if v == parent:
                    continue
                result += dfs(n, lookup, graph, u, v)
            return result
        (lookup, graph) = (set(), collections.defaultdict(list))
        for (u, v) in connections:
            lookup.add(u * n + v)
            graph[v].append(u)
            graph[u].append(v)
        return dfs(n, lookup, graph, -1, 0)