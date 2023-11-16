import collections

class Solution(object):

    def treeDiameter(self, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        (graph, length) = (collections.defaultdict(set), 0)
        for (u, v) in edges:
            graph[u].add(v)
            graph[v].add(u)
        curr_level = {(None, u) for (u, neighbors) in graph.iteritems() if len(neighbors) == 1}
        while curr_level:
            curr_level = {(u, v) for (prev, u) in curr_level for v in graph[u] if v != prev}
            length += 1
        return max(length - 1, 0)