class Solution(object):

    def findJudge(self, N, trust):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :type trust: List[List[int]]\n        :rtype: int\n        '
        degrees = [0] * N
        for (i, j) in trust:
            degrees[i - 1] -= 1
            degrees[j - 1] += 1
        for i in xrange(len(degrees)):
            if degrees[i] == N - 1:
                return i + 1
        return -1