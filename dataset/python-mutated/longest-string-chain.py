import collections

class Solution(object):

    def longestStrChain(self, words):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :rtype: int\n        '
        words.sort(key=len)
        dp = collections.defaultdict(int)
        for w in words:
            for i in xrange(len(w)):
                dp[w] = max(dp[w], dp[w[:i] + w[i + 1:]] + 1)
        return max(dp.itervalues())