class Solution(object):

    def minimumTime(self, time, totalTrips):
        if False:
            return 10
        '\n        :type time: List[int]\n        :type totalTrips: int\n        :rtype: int\n        '

        def check(time, totalTrips, x):
            if False:
                return 10
            return sum((x // t for t in time)) >= totalTrips
        (left, right) = (1, max(time) * totalTrips)
        while left <= right:
            mid = left + (right - left) // 2
            if check(time, totalTrips, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left