import collections

class Solution(object):

    def minOperations(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b - 1) // b
        cnt = collections.Counter(nums)
        return sum((ceil_divide(x, 3) for x in cnt.itervalues())) if all((x >= 2 for x in cnt.itervalues())) else -1