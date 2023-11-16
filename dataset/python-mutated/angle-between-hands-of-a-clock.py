class Solution(object):

    def angleClock(self, hour, minutes):
        if False:
            print('Hello World!')
        '\n        :type hour: int\n        :type minutes: int\n        :rtype: float\n        '
        angle1 = (hour % 12 * 60.0 + minutes) / 720.0
        angle2 = minutes / 60.0
        diff = abs(angle1 - angle2)
        return min(diff, 1.0 - diff) * 360.0