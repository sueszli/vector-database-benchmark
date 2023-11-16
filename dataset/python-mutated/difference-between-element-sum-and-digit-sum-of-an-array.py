class Solution(object):

    def differenceOfSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def total(x):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            while x:
                result += x % 10
                x //= 10
            return result
        return abs(sum(nums) - sum((total(x) for x in nums)))