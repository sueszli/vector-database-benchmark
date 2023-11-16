class Solution(object):

    def maxPower(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (result, count) = (1, 1)
        for i in xrange(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                count = 1
            result = max(result, count)
        return result
import itertools

class Solution2(object):

    def maxPower(self, s):
        if False:
            while True:
                i = 10
        return max((len(list(v)) for (_, v) in itertools.groupby(s)))