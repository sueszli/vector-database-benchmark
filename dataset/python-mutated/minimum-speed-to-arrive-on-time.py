class Solution(object):

    def minSpeedOnTime(self, dist, hour):
        if False:
            while True:
                i = 10
        '\n        :type dist: List[int]\n        :type hour: float\n        :rtype: int\n        '

        def ceil(a, b):
            if False:
                print('Hello World!')
            return (a + b - 1) // b

        def total_time(dist, x):
            if False:
                return 10
            return sum((ceil(dist[i], x) for i in xrange(len(dist) - 1))) + float(dist[-1]) / x

        def check(dist, hour, x):
            if False:
                while True:
                    i = 10
            return total_time(dist, x) <= hour
        MAX_SPEED = 10 ** 7
        if not check(dist, hour, MAX_SPEED):
            return -1
        (left, right) = (1, MAX_SPEED)
        while left <= right:
            mid = left + (right - left) // 2
            if check(dist, hour, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left