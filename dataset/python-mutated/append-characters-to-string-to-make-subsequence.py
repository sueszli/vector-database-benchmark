class Solution(object):

    def appendCharacters(self, s, t):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type t: str\n        :rtype: int\n        '
        i = -1
        for (j, c) in enumerate(t):
            for i in xrange(i + 1, len(s)):
                if s[i] == c:
                    break
            else:
                return len(t) - j
        return 0