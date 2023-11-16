import operator
import collections

class Solution(object):

    def singleNumber(self, nums):
        if False:
            i = 10
            return i + 15
        x_xor_y = reduce(operator.xor, nums)
        bit = x_xor_y & -x_xor_y
        result = [0, 0]
        for i in nums:
            result[bool(i & bit)] ^= i
        return result

class Solution2(object):

    def singleNumber(self, nums):
        if False:
            print('Hello World!')
        x_xor_y = 0
        for i in nums:
            x_xor_y ^= i
        bit = x_xor_y & ~(x_xor_y - 1)
        x = 0
        for i in nums:
            if i & bit:
                x ^= i
        return [x, x ^ x_xor_y]

class Solution3(object):

    def singleNumber(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [x[0] for x in sorted(collections.Counter(nums).items(), key=lambda i: i[1], reverse=False)[:2]]