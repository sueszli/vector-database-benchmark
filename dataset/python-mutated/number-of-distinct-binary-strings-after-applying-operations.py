class Solution(object):

    def countDistinctStrings(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        return pow(2, len(s) - k + 1, MOD)