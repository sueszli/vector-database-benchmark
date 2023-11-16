class Solution(object):

    def averageValue(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        total = cnt = 0
        for x in nums:
            if x % 6:
                continue
            total += x
            cnt += 1
        return total // cnt if cnt else 0