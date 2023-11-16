class Solution(object):

    def longestAlternatingSubarray(self, nums, threshold):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type threshold: int\n        :rtype: int\n        '
        result = l = 0
        for x in nums:
            if x > threshold:
                l = 0
                continue
            if l % 2 == x % 2:
                l += 1
            else:
                l = int(x % 2 == 0)
            result = max(result, l)
        return result