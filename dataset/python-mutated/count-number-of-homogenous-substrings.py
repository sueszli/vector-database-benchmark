class Solution(object):

    def countHomogenous(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = cnt = 0
        for i in xrange(len(s)):
            if i and s[i - 1] == s[i]:
                cnt += 1
            else:
                cnt = 1
            result = (result + cnt) % MOD
        return result