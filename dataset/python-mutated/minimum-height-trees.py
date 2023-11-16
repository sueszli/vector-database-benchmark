import collections

class Solution(object):

    def findMinHeightTrees(self, n, edges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '
        if n == 1:
            return [0]
        neighbors = collections.defaultdict(set)
        for (u, v) in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        (pre_level, unvisited) = ([], set())
        for i in xrange(n):
            if len(neighbors[i]) == 1:
                pre_level.append(i)
            unvisited.add(i)
        while len(unvisited) > 2:
            cur_level = []
            for u in pre_level:
                unvisited.remove(u)
                for v in neighbors[u]:
                    if v in unvisited:
                        neighbors[v].remove(u)
                        if len(neighbors[v]) == 1:
                            cur_level.append(v)
            pre_level = cur_level
        return list(unvisited)