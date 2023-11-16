import heapq

class Solution(object):

    def convertArray(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def f(nums):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            max_heap = []
            for x in nums:
                if max_heap and x < -max_heap[0]:
                    result += -heapq.heappop(max_heap) - x
                    heapq.heappush(max_heap, -x)
                heapq.heappush(max_heap, -x)
            return result
        return min(f(nums), f((x for x in reversed(nums))))
import collections

class Solution2(object):

    def convertArray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        vals = sorted(set(nums))

        def f(nums):
            if False:
                while True:
                    i = 10
            dp = collections.defaultdict(int)
            for x in nums:
                prev = -1
                for i in vals:
                    dp[i] = min(dp[i] + abs(i - x), dp[prev]) if prev != -1 else dp[i] + abs(i - x)
                    prev = i
            return dp[vals[-1]]
        return min(f(nums), f((x for x in reversed(nums))))