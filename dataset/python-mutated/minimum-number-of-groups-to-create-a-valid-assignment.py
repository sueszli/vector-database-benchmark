import collections

class Solution(object):

    def minGroupsForValidAssignment(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        INF = float('inf')

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b - 1) // b

        def count(x):
            if False:
                print('Hello World!')
            result = 0
            for c in cnt.itervalues():
                if c % x > c // x:
                    return INF
                result += ceil_divide(c, x + 1)
            return result
        cnt = collections.Counter(nums)
        for i in reversed(xrange(1, min(cnt.itervalues()) + 1)):
            c = count(i)
            if c != INF:
                return c
        return 0