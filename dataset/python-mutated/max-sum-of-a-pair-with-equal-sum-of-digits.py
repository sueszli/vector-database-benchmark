class Solution(object):

    def maximumSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def sum_digits(x):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            while x:
                result += x % 10
                x //= 10
            return result
        lookup = {}
        result = -1
        for x in nums:
            k = sum_digits(x)
            if k not in lookup:
                lookup[k] = x
                continue
            result = max(result, lookup[k] + x)
            if x > lookup[k]:
                lookup[k] = x
        return result