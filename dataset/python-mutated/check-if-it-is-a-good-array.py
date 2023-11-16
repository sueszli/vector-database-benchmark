class Solution(object):

    def isGoodArray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: bool\n        '

        def gcd(a, b):
            if False:
                return 10
            while b:
                (a, b) = (b, a % b)
            return a
        result = nums[0]
        for num in nums:
            result = gcd(result, num)
            if result == 1:
                break
        return result == 1