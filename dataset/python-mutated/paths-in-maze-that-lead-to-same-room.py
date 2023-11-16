class Solution(object):

    def numberOfPaths(self, n, corridors):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type corridors: List[List[int]]\n        :rtype: int\n        '
        adj = [set() for _ in xrange(n)]
        for (u, v) in corridors:
            adj[min(u, v) - 1].add(max(u, v) - 1)
        return sum((k in adj[i] for i in xrange(n) for j in adj[i] for k in adj[j]))