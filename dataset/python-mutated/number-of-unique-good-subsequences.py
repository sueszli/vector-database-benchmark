class Solution(object):

    def numberOfUniqueGoodSubsequences(self, binary):
        if False:
            print('Hello World!')
        '\n        :type binary: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (ends0, ends1) = (0, 0)
        has_zero = False
        for b in binary:
            if b == '1':
                ends1 = (ends0 + ends1 + 1) % MOD
            else:
                ends0 = (ends0 + ends1) % MOD
                has_zero = True
        return (ends0 + ends1 + int(has_zero)) % MOD