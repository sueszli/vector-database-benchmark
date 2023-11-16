class Solution(object):

    def findDuplicates(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = []
        for i in nums:
            if nums[abs(i) - 1] < 0:
                result.append(abs(i))
            else:
                nums[abs(i) - 1] *= -1
        return result

class Solution2(object):

    def findDuplicates(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = []
        i = 0
        while i < len(nums):
            if nums[i] != nums[nums[i] - 1]:
                (nums[nums[i] - 1], nums[i]) = (nums[i], nums[nums[i] - 1])
            else:
                i += 1
        for i in xrange(len(nums)):
            if i != nums[i] - 1:
                result.append(nums[i])
        return result
from collections import Counter

class Solution3(object):

    def findDuplicates(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [elem for (elem, count) in Counter(nums).items() if count == 2]