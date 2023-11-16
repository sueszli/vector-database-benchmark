class Solution(object):

    def checkSubarraySum(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '
        count = 0
        lookup = {0: -1}
        for (i, num) in enumerate(nums):
            count += num
            if k:
                count %= k
            if count in lookup:
                if i - lookup[count] > 1:
                    return True
            else:
                lookup[count] = i
        return False