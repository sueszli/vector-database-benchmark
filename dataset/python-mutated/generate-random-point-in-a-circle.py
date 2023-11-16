import random
import math

class Solution(object):

    def __init__(self, radius, x_center, y_center):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type radius: float\n        :type x_center: float\n        :type y_center: float\n        '
        self.__radius = radius
        self.__x_center = x_center
        self.__y_center = y_center

    def randPoint(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: List[float]\n        '
        r = self.__radius * math.sqrt(random.uniform(0, 1))
        theta = 2 * math.pi * random.uniform(0, 1)
        return (r * math.cos(theta) + self.__x_center, r * math.sin(theta) + self.__y_center)