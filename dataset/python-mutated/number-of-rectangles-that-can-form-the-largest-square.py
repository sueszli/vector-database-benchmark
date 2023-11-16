class Solution(object):

    def countGoodRectangles(self, rectangles):
        if False:
            print('Hello World!')
        '\n        :type rectangles: List[List[int]]\n        :rtype: int\n        '
        result = mx = 0
        for (l, w) in rectangles:
            side = min(l, w)
            if side > mx:
                (result, mx) = (1, side)
            elif side == mx:
                result += 1
        return result