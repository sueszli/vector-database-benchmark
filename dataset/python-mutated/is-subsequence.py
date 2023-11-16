class Solution(object):

    def isSubsequence(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        if not s:
            return True
        i = 0
        for c in t:
            if c == s[i]:
                i += 1
            if i == len(s):
                break
        return i == len(s)