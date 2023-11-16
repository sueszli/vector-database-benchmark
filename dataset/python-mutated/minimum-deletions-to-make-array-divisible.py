class Solution(object):

    def minOperations(self, nums, numsDivide):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type numsDivide: List[int]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                i = 10
                return i + 15
            while b:
                (a, b) = (b, a % b)
            return a
        g = reduce(gcd, numsDivide)
        mn = float('inf')
        for x in nums:
            if g % x == 0:
                mn = min(mn, x)
        return sum((x < mn for x in nums)) if mn != float('inf') else -1