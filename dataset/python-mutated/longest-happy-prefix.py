class Solution(object):

    def longestPrefix(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '

        def getPrefix(pattern):
            if False:
                i = 10
                return i + 15
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j != -1 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix
        return s[:getPrefix(s)[-1] + 1]

class Solution2(object):

    def longestPrefix(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        M = 10 ** 9 + 7
        D = 26

        def check(l, s):
            if False:
                i = 10
                return i + 15
            for i in xrange(l):
                if s[i] != s[len(s) - l + i]:
                    return False
            return True
        (result, prefix, suffix, power) = (0, 0, 0, 1)
        for i in xrange(len(s) - 1):
            prefix = (prefix * D + (ord(s[i]) - ord('a'))) % M
            suffix = (suffix + (ord(s[len(s) - (i + 1)]) - ord('a')) * power) % M
            power = power * D % M
            if prefix == suffix:
                result = i + 1
        return s[:result]