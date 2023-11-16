class Solution(object):

    def findMaxK(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = set(nums)
        return max([x for x in lookup if x > 0 and -x in lookup] or [-1])