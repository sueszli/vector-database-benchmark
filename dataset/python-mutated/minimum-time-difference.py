class Solution(object):

    def findMinDifference(self, timePoints):
        if False:
            i = 10
            return i + 15
        '\n        :type timePoints: List[str]\n        :rtype: int\n        '
        minutes = map(lambda x: int(x[:2]) * 60 + int(x[3:]), timePoints)
        minutes.sort()
        return min(((y - x) % (24 * 60) for (x, y) in zip(minutes, minutes[1:] + minutes[:1])))