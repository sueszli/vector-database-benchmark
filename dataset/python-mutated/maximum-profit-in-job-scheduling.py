import itertools
import bisect

class Solution(object):

    def jobScheduling(self, startTime, endTime, profit):
        if False:
            return 10
        '\n        :type startTime: List[int]\n        :type endTime: List[int]\n        :type profit: List[int]\n        :rtype: int\n        '
        jobs = sorted(itertools.izip(endTime, startTime, profit))
        dp = [(0, 0)]
        for (e, s, p) in jobs:
            i = bisect.bisect_right(dp, (s + 1, 0)) - 1
            if dp[i][1] + p > dp[-1][1]:
                dp.append((e, dp[i][1] + p))
        return dp[-1][1]
import heapq

class Solution(object):

    def jobScheduling(self, startTime, endTime, profit):
        if False:
            print('Hello World!')
        '\n        :type startTime: List[int]\n        :type endTime: List[int]\n        :type profit: List[int]\n        :rtype: int\n        '
        min_heap = zip(startTime, endTime, profit)
        heapq.heapify(min_heap)
        result = 0
        while min_heap:
            (s, e, p) = heapq.heappop(min_heap)
            if s < e:
                heapq.heappush(min_heap, (e, s, result + p))
            else:
                result = max(result, p)
        return result