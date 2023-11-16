class Solution(object):

    def isOneEditDistance(self, s, t):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type t: str\n        :rtype: bool\n        '
        (m, n) = (len(s), len(t))
        if m > n:
            return self.isOneEditDistance(t, s)
        if n - m > 1:
            return False
        (i, shift) = (0, n - m)
        while i < m and s[i] == t[i]:
            i += 1
        if shift == 0:
            i += 1
        while i < m and s[i] == t[i + shift]:
            i += 1
        return i == m