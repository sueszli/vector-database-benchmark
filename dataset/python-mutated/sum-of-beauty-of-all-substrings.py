class Solution(object):

    def beautySum(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = 0
        for i in xrange(len(s)):
            lookup = [0] * 26
            for j in xrange(i, len(s)):
                lookup[ord(s[j]) - ord('a')] += 1
                result += max(lookup) - min((x for x in lookup if x))
        return result