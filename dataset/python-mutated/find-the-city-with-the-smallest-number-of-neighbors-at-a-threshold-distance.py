class Solution(object):

    def findTheCity(self, n, edges, distanceThreshold):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type distanceThreshold: int\n        :rtype: int\n        '
        dist = [[float('inf')] * n for _ in xrange(n)]
        for (i, j, w) in edges:
            dist[i][j] = dist[j][i] = w
        for i in xrange(n):
            dist[i][i] = 0
        for k in xrange(n):
            for i in xrange(n):
                for j in xrange(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        result = {sum((d <= distanceThreshold for d in dist[i])): i for i in xrange(n)}
        return result[min(result.iterkeys())]