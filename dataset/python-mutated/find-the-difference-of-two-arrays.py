class Solution(object):

    def findDifference(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[List[int]]\n        '
        lookup = [set(nums1), set(nums2)]
        return [list(lookup[0] - lookup[1]), list(lookup[1] - lookup[0])]