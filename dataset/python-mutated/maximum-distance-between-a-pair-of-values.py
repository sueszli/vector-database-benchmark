class Solution(object):

    def maxDistance(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        result = i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] > nums2[j]:
                i += 1
            else:
                result = max(result, j - i)
                j += 1
        return result