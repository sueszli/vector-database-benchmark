class Solution(object):

    def mergeArrays(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[List[int]]\n        :type nums2: List[List[int]]\n        :rtype: List[List[int]]\n        '
        result = []
        i = j = 0
        while i < len(nums1) or j < len(nums2):
            if j == len(nums2) or (i < len(nums1) and nums1[i][0] < nums2[j][0]):
                if result and result[-1][0] == nums1[i][0]:
                    result[-1][1] += nums1[i][1]
                else:
                    result.append(nums1[i])
                i += 1
            else:
                if result and result[-1][0] == nums2[j][0]:
                    result[-1][1] += nums2[j][1]
                else:
                    result.append(nums2[j])
                j += 1
        return result