class Solution(object):

    def sortJumbled(self, mapping, nums):
        if False:
            while True:
                i = 10
        '\n        :type mapping: List[int]\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def transform(mapping, x):
            if False:
                for i in range(10):
                    print('nop')
            if not x:
                return mapping[x]
            (result, base) = (0, 1)
            while x:
                result += mapping[x % 10] * base
                x //= 10
                base *= 10
            return result
        return [nums[i] for (_, i) in sorted(((transform(mapping, nums[i]), i) for i in xrange(len(nums))))]