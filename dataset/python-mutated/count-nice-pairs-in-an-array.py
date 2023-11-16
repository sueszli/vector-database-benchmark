import collections

class Solution(object):

    def countNicePairs(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def rev(x):
            if False:
                print('Hello World!')
            result = 0
            while x:
                (x, r) = divmod(x, 10)
                result = result * 10 + r
            return result
        result = 0
        lookup = collections.defaultdict(int)
        for num in nums:
            result = (result + lookup[num - rev(num)]) % MOD
            lookup[num - rev(num)] += 1
        return result