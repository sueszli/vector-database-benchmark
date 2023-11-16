import collections

class Solution(object):

    def maxOperations(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        count = collections.Counter()
        result = 0
        for x in nums:
            if k - x in count and count[k - x]:
                count[k - x] -= 1
                result += 1
            else:
                count[x] += 1
        return result