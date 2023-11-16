class Solution(object):

    def checkOverlap(self, radius, x_center, y_center, x1, y1, x2, y2):
        if False:
            print('Hello World!')
        '\n        :type radius: int\n        :type x_center: int\n        :type y_center: int\n        :type x1: int\n        :type y1: int\n        :type x2: int\n        :type y2: int\n        :rtype: bool\n        '
        x1 -= x_center
        y1 -= y_center
        x2 -= x_center
        y2 -= y_center
        x = x1 if x1 > 0 else x2 if x2 < 0 else 0
        y = y1 if y1 > 0 else y2 if y2 < 0 else 0
        return x ** 2 + y ** 2 <= radius ** 2

class Solution2(object):

    def checkOverlap(self, radius, x_center, y_center, x1, y1, x2, y2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type radius: int\n        :type x_center: int\n        :type y_center: int\n        :type x1: int\n        :type y1: int\n        :type x2: int\n        :type y2: int\n        :rtype: bool\n        '
        x1 -= x_center
        y1 -= y_center
        x2 -= x_center
        y2 -= y_center
        x = min(abs(x1), abs(x2)) if x1 * x2 > 0 else 0
        y = min(abs(y1), abs(y2)) if y1 * y2 > 0 else 0
        return x ** 2 + y ** 2 <= radius ** 2