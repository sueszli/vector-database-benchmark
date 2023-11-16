class Solution(object):

    def minOperations(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                i = 10
                return i + 15
            while b:
                (a, b) = (b, a % b)
            return a
        cnt = nums.count(1)
        if cnt:
            return len(nums) - cnt
        result = float('inf')
        for i in xrange(len(nums)):
            g = nums[i]
            for j in range(i + 1, len(nums)):
                g = gcd(g, nums[j])
                if g == 1:
                    result = min(result, j - i)
                    break
        return result + (len(nums) - 1) if result != float('inf') else -1