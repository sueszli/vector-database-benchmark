class Solution(object):

    def maximumGain(self, s, x, y):
        if False:
            return 10
        '\n        :type s: str\n        :type x: int\n        :type y: int\n        :rtype: int\n        '

        def score(s, a, x):
            if False:
                i = 10
                return i + 15
            i = result = 0
            for j in xrange(len(s)):
                s[i] = s[j]
                i += 1
                if i >= 2 and s[i - 2:i] == a:
                    i -= 2
                    result += x
            s[:] = s[:i]
            return result
        (s, a, b) = (list(s), list('ab'), list('ba'))
        if x < y:
            (x, y) = (y, x)
            (a, b) = (b, a)
        return score(s, a, x) + score(s, b, y)