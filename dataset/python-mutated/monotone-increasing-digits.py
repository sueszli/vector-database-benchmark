class Solution(object):

    def monotoneIncreasingDigits(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: int\n        '
        nums = map(int, list(str(N)))
        leftmost_inverted_idx = len(nums)
        for i in reversed(xrange(1, len(nums))):
            if nums[i - 1] > nums[i]:
                leftmost_inverted_idx = i
                nums[i - 1] -= 1
        for i in xrange(leftmost_inverted_idx, len(nums)):
            nums[i] = 9
        return int(''.join(map(str, nums)))