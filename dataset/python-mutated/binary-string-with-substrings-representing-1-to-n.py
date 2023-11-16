class Solution(object):

    def queryString(self, S, N):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :type N: int\n        :rtype: bool\n        '
        return all((bin(i)[2:] in S for i in reversed(xrange(N // 2, N + 1))))