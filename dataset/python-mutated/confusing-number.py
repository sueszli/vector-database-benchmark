class Solution(object):

    def confusingNumber(self, N):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :rtype: bool\n        '
        lookup = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
        S = str(N)
        result = []
        for i in xrange(len(S)):
            if S[i] not in lookup:
                return False
        for i in xrange((len(S) + 1) // 2):
            if S[i] != lookup[S[-(i + 1)]]:
                return True
        return False