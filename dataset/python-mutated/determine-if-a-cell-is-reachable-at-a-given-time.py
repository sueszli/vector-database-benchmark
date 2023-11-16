class Solution(object):

    def isReachableAtTime(self, sx, sy, fx, fy, t):
        if False:
            return 10
        '\n        :type sx: int\n        :type sy: int\n        :type fx: int\n        :type fy: int\n        :type t: int\n        :rtype: bool\n        '
        (diff1, diff2) = (abs(sx - fx), abs(sy - fy))
        mn = min(diff1, diff2) + abs(diff1 - diff2)
        return t >= mn if mn else t != 1