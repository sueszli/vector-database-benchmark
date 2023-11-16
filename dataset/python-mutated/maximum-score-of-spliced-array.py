class Solution(object):

    def maximumsSplicedArray(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '

        def kadane(a):
            if False:
                return 10
            result = curr = 0
            for x in a:
                curr = max(curr + x, 0)
                result = max(result, curr)
            return result
        return max(sum(nums1) + kadane((nums2[i] - nums1[i] for i in xrange(len(nums1)))), sum(nums2) + kadane((nums1[i] - nums2[i] for i in xrange(len(nums2)))))