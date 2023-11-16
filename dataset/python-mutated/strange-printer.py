class Solution(object):

    def strangePrinter(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '

        def dp(s, i, j, lookup):
            if False:
                i = 10
                return i + 15
            if i > j:
                return 0
            if (i, j) not in lookup:
                lookup[i, j] = dp(s, i, j - 1, lookup) + 1
                for k in xrange(i, j):
                    if s[k] == s[j]:
                        lookup[i, j] = min(lookup[i, j], dp(s, i, k, lookup) + dp(s, k + 1, j - 1, lookup))
            return lookup[i, j]
        lookup = {}
        return dp(s, 0, len(s) - 1, lookup)