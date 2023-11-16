"""
Topic: 通过字符串调用方法
Desc : 
"""
import math

class Point:

    def __init__(self, x, y):
        if False:
            return 10
        self.x = x
        self.y = y

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Point({!r:},{!r:})'.format(self.x, self.y)

    def distance(self, x, y):
        if False:
            i = 10
            return i + 15
        return math.hypot(self.x - x, self.y - y)
p = Point(2, 3)
d = getattr(p, 'distance')(0, 0)
import operator
operator.methodcaller('distance', 0, 0)(p)
points = [Point(1, 2), Point(3, 0), Point(10, -3), Point(-5, -7), Point(-1, 8), Point(3, 2)]
points.sort(key=operator.methodcaller('distance', 0, 0))