class Solution(object):

    def minmaxGasDist(self, stations, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type stations: List[int]\n        :type K: int\n        :rtype: float\n        '

        def possible(stations, K, guess):
            if False:
                i = 10
                return i + 15
            return sum((int((stations[i + 1] - stations[i]) / guess) for i in xrange(len(stations) - 1))) <= K
        (left, right) = (0, 10 ** 8)
        while right - left > 1e-06:
            mid = left + (right - left) / 2.0
            if possible(mid):
                right = mid
            else:
                left = mid
        return left