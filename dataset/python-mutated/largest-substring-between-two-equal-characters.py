class Solution(object):

    def maxLengthBetweenEqualCharacters(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (result, lookup) = (-1, {})
        for (i, c) in enumerate(s):
            result = max(result, i - lookup.setdefault(c, i) - 1)
        return result