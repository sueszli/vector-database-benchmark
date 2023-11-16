class Solution(object):

    def edgeScore(self, edges):
        if False:
            print('Hello World!')
        '\n        :type edges: List[int]\n        :rtype: int\n        '
        score = [0] * len(edges)
        for (u, v) in enumerate(edges):
            score[v] += u
        return max(xrange(len(edges)), key=lambda x: score[x])