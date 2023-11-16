class Solution(object):

    def findPoisonedDuration(self, timeSeries, duration):
        if False:
            while True:
                i = 10
        '\n        :type timeSeries: List[int]\n        :type duration: int\n        :rtype: int\n        '
        result = duration * len(timeSeries)
        for i in xrange(1, len(timeSeries)):
            result -= max(0, duration - (timeSeries[i] - timeSeries[i - 1]))
        return result