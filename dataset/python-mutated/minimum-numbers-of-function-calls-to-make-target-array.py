class Solution(object):

    def minOperations(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def popcount(n):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        (result, max_len) = (0, 1)
        for num in nums:
            result += popcount(num)
            max_len = max(max_len, num.bit_length())
        return result + (max_len - 1)