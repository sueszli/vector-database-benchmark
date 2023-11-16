class Solution(object):

    def makeFancyString(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        s = list(s)
        cnt = j = 0
        for (i, c) in enumerate(s):
            cnt = cnt + 1 if i >= 1 and c == s[i - 1] else 1
            if cnt < 3:
                s[j] = c
                j += 1
        s[:] = s[:j]
        return ''.join(s)