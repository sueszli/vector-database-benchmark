import collections

class Solution(object):

    def longestSubarray(self, nums, limit):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type limit: int\n        :rtype: int\n        '
        (max_dq, min_dq) = (collections.deque(), collections.deque())
        left = 0
        for (right, num) in enumerate(nums):
            while max_dq and nums[max_dq[-1]] <= num:
                max_dq.pop()
            max_dq.append(right)
            while min_dq and nums[min_dq[-1]] >= num:
                min_dq.pop()
            min_dq.append(right)
            if nums[max_dq[0]] - nums[min_dq[0]] > limit:
                if max_dq[0] == left:
                    max_dq.popleft()
                if min_dq[0] == left:
                    min_dq.popleft()
                left += 1
        return len(nums) - left
import collections

class Solution2(object):

    def longestSubarray(self, nums, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type limit: int\n        :rtype: int\n        '
        (max_dq, min_dq) = (collections.deque(), collections.deque())
        (result, left) = (0, 0)
        for (right, num) in enumerate(nums):
            while max_dq and nums[max_dq[-1]] <= num:
                max_dq.pop()
            max_dq.append(right)
            while min_dq and nums[min_dq[-1]] >= num:
                min_dq.pop()
            min_dq.append(right)
            while nums[max_dq[0]] - nums[min_dq[0]] > limit:
                if max_dq[0] == left:
                    max_dq.popleft()
                if min_dq[0] == left:
                    min_dq.popleft()
                left += 1
            result = max(result, right - left + 1)
        return result