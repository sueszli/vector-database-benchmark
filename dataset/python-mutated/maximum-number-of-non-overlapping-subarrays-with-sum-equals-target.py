class Solution(object):

    def maxNonOverlapping(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        lookup = {0: -1}
        (result, accu, right) = (0, 0, -1)
        for (i, num) in enumerate(nums):
            accu += num
            if accu - target in lookup and lookup[accu - target] >= right:
                right = i
                result += 1
            lookup[accu] = i
        return result