import collections

class Solution(object):

    def continuousSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = left = 0
        (mn, mx) = (float('inf'), float('-inf'))
        for right in xrange(len(nums)):
            if mn <= nums[right] <= mx:
                (mn, mx) = (max(mn, nums[right] - 2), min(mx, nums[right] + 2))
            else:
                (mn, mx) = (nums[right] - 2, nums[right] + 2)
                for left in reversed(xrange(right)):
                    if not mn <= nums[left] <= mx:
                        break
                    (mn, mx) = (max(mn, nums[left] - 2), min(mx, nums[left] + 2))
                else:
                    left = -1
                left += 1
            result += right - left + 1
        return result
import collections

class Solution2(object):

    def continuousSubarrays(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (mn, mx) = (collections.deque(), collections.deque())
        result = left = 0
        for right in xrange(len(nums)):
            while mn and nums[mn[-1]] > nums[right]:
                mn.pop()
            mn.append(right)
            while mx and nums[mx[-1]] < nums[right]:
                mx.pop()
            mx.append(right)
            while not nums[right] - nums[mn[0]] <= 2:
                left = max(left, mn.popleft() + 1)
            while not nums[mx[0]] - nums[right] <= 2:
                left = max(left, mx.popleft() + 1)
            result += right - left + 1
        return result
from sortedcontainers import SortedDict

class Solution3(object):

    def continuousSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = left = 0
        lookup = SortedDict()
        for right in xrange(len(nums)):
            lookup[nums[right]] = right
            to_del = []
            for (x, i) in lookup.items():
                if nums[right] - x <= 2:
                    break
                left = max(left, i + 1)
                to_del.append(x)
            for (x, i) in reversed(lookup.items()):
                if x - nums[right] <= 2:
                    break
                left = max(left, i + 1)
                to_del.append(x)
            for x in to_del:
                del lookup[x]
            result += right - left + 1
        return result