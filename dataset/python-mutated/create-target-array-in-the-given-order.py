class Solution(object):

    def createTargetArray(self, nums, index):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type index: List[int]\n        :rtype: List[int]\n        '
        for i in xrange(len(nums)):
            for j in xrange(i):
                if index[j] >= index[i]:
                    index[j] += 1
        result = [0] * len(nums)
        for i in xrange(len(nums)):
            result[index[i]] = nums[i]
        return result
import itertools

class Solution2(object):

    def createTargetArray(self, nums, index):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type index: List[int]\n        :rtype: List[int]\n        '
        result = []
        for (i, x) in itertools.izip(index, nums):
            result.insert(i, x)
        return result