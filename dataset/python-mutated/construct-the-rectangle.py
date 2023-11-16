import math

class Solution(object):

    def constructRectangle(self, area):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type area: int\n        :rtype: List[int]\n        '
        w = int(math.sqrt(area))
        while area % w:
            w -= 1
        return [area // w, w]