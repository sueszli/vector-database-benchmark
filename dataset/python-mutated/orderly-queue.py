class Solution(object):

    def orderlyQueue(self, S, K):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :type K: int\n        :rtype: str\n        '
        if K == 1:
            return min((S[i:] + S[:i] for i in xrange(len(S))))
        return ''.join(sorted(S))