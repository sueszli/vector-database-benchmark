class Solution(object):

    def subStrHash(self, s, power, modulo, k, hashValue):
        if False:
            return 10
        '\n        :type s: str\n        :type power: int\n        :type modulo: int\n        :type k: int\n        :type hashValue: int\n        :rtype: str\n        '
        (h, idx) = (0, -1)
        pw = pow(power, k - 1, modulo)
        for i in reversed(xrange(len(s))):
            if i + k < len(s):
                h = (h - (ord(s[i + k]) - ord('a') + 1) * pw) % modulo
            h = (h * power + (ord(s[i]) - ord('a') + 1)) % modulo
            if h == hashValue:
                idx = i
        return s[idx:idx + k]