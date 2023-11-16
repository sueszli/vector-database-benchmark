class Solution(object):

    def findMiddleIndex(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        total = sum(nums)
        accu = 0
        for (i, x) in enumerate(nums):
            if accu * 2 == total - x:
                return i
            accu += x
        return -1