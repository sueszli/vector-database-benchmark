import heapq

class Solution(object):

    def maxRunTime(self, n, batteries):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type batteries: List[int]\n        :rtype: int\n        '
        total = sum(batteries)
        for i in xrange(len(batteries)):
            batteries[i] = -batteries[i]
        heapq.heapify(batteries)
        while -batteries[0] > total // n:
            n -= 1
            total -= -heapq.heappop(batteries)
        return total // n

class Solution2(object):

    def maxRunTime(self, n, batteries):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type batteries: List[int]\n        :rtype: int\n        '

        def check(n, batteries, x):
            if False:
                return 10
            return sum((min(b, x) for b in batteries)) >= n * x
        (left, right) = (min(batteries), sum(batteries) // n)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(n, batteries, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right