import collections

class Solution(object):

    def maxResult(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        score = 0
        dq = collections.deque()
        for (i, num) in enumerate(nums):
            if dq and dq[0][0] == i - k - 1:
                dq.popleft()
            score = num if not dq else dq[0][1] + num
            while dq and dq[-1][1] <= score:
                dq.pop()
            dq.append((i, score))
        return score