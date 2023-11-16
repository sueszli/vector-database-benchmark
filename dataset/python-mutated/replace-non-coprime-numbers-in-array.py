class Solution(object):

    def replaceNonCoprimes(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def gcd(a, b):
            if False:
                return 10
            while b:
                (a, b) = (b, a % b)
            return a
        result = []
        for x in nums:
            while True:
                g = gcd(result[-1] if result else 1, x)
                if g == 1:
                    break
                x *= result.pop() // g
            result.append(x)
        return result