import collections

class Solution(object):

    def constrainedSubsetSum(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        (result, dq) = (float('-inf'), collections.deque())
        for i in xrange(len(nums)):
            if dq and i - dq[0][0] == k + 1:
                dq.popleft()
            curr = nums[i] + (dq[0][1] if dq else 0)
            while dq and dq[-1][1] <= curr:
                dq.pop()
            if curr > 0:
                dq.append((i, curr))
            result = max(result, curr)
        return result