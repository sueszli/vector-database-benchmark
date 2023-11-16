class Solution(object):

    def numSub(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (result, count) = (0, 0)
        for c in s:
            count = count + 1 if c == '1' else 0
            result = (result + count) % MOD
        return result