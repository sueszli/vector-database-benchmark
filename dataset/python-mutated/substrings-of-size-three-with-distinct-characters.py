import collections

class Solution(object):

    def countGoodSubstrings(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        K = 3
        result = 0
        count = collections.Counter()
        for i in xrange(len(s)):
            if i >= K:
                count[s[i - K]] -= 1
                if not count[s[i - K]]:
                    del count[s[i - K]]
            count[s[i]] += 1
            if len(count) == K:
                result += 1
        return result