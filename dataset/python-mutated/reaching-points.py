class Solution(object):

    def reachingPoints(self, sx, sy, tx, ty):
        if False:
            print('Hello World!')
        '\n        :type sx: int\n        :type sy: int\n        :type tx: int\n        :type ty: int\n        :rtype: bool\n        '
        while tx >= sx and ty >= sy:
            if tx < ty:
                (sx, sy) = (sy, sx)
                (tx, ty) = (ty, tx)
            if ty > sy:
                tx %= ty
            else:
                return (tx - sx) % ty == 0
        return False