class Solution(object):

    def sumIndicesWithKSetBits(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def next_popcount(n):
            if False:
                return 10
            lowest_bit = n & -n
            left_bits = n + lowest_bit
            changed_bits = n ^ left_bits
            right_bits = changed_bits // lowest_bit >> 2
            return left_bits | right_bits
        result = 0
        i = (1 << k) - 1
        while i < len(nums):
            result += nums[i]
            if i == 0:
                break
            i = next_popcount(i)
        return result

class Solution2(object):

    def sumIndicesWithKSetBits(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                for i in range(10):
                    print('nop')
            return bin(x)[1:].count('1')
        return sum((x for (i, x) in enumerate(nums) if popcount(i) == k))