class Solution(object):

    def totalHammingDistance(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for i in xrange(32):
            counts = [0] * 2
            for num in nums:
                counts[num >> i & 1] += 1
            result += counts[0] * counts[1]
        return result