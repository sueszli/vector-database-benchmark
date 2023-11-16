class Solution(object):

    def maxSum(self, nums1, nums2):
        if False:
            while True:
                i = 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (i, j) = (0, 0)
        (result, sum1, sum2) = (0, 0, 0)
        while i != len(nums1) or j != len(nums2):
            if i != len(nums1) and (j == len(nums2) or nums1[i] < nums2[j]):
                sum1 += nums1[i]
                i += 1
            elif j != len(nums2) and (i == len(nums1) or nums1[i] > nums2[j]):
                sum2 += nums2[j]
                j += 1
            else:
                result = (result + (max(sum1, sum2) + nums1[i])) % MOD
                (sum1, sum2) = (0, 0)
                i += 1
                j += 1
        return (result + max(sum1, sum2)) % MOD