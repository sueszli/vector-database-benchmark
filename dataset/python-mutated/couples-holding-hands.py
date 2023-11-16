class Solution(object):

    def minSwapsCouples(self, row):
        if False:
            while True:
                i = 10
        '\n        :type row: List[int]\n        :rtype: int\n        '
        N = len(row) // 2
        couples = [[] for _ in xrange(N)]
        for (seat, num) in enumerate(row):
            couples[num // 2].append(seat // 2)
        adj = [[] for _ in xrange(N)]
        for (couch1, couch2) in couples:
            adj[couch1].append(couch2)
            adj[couch2].append(couch1)
        result = 0
        for couch in xrange(N):
            if not adj[couch]:
                continue
            (couch1, couch2) = (couch, adj[couch].pop())
            while couch2 != couch:
                result += 1
                adj[couch2].remove(couch1)
                (couch1, couch2) = (couch2, adj[couch2].pop())
        return result