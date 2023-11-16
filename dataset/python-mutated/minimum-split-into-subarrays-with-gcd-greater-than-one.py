class Solution(object):

    def minimumSplits(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                print('Hello World!')
            while b:
                (a, b) = (b, a % b)
            return a
        (result, g) = (1, 0)
        for x in nums:
            g = gcd(g, x)
            if g == 1:
                g = x
                result += 1
        return result