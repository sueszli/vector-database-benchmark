import collections

class DetectSquares(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__x_to_ys = collections.defaultdict(set)
        self.__point_counts = collections.defaultdict(int)

    def add(self, point):
        if False:
            i = 10
            return i + 15
        '\n        :type point: List[int]\n        :rtype: None\n        '
        self.__x_to_ys[point[0]].add(point[1])
        self.__point_counts[tuple(point)] += 1

    def count(self, point):
        if False:
            while True:
                i = 10
        '\n        :type point: List[int]\n        :rtype: int\n        '
        result = 0
        for y in self.__x_to_ys[point[0]]:
            if y == point[1]:
                continue
            dy = y - point[1]
            result += self.__point_counts[point[0], y] * self.__point_counts[point[0] + dy, point[1]] * self.__point_counts[point[0] + dy, y]
            result += self.__point_counts[point[0], y] * self.__point_counts[point[0] - dy, point[1]] * self.__point_counts[point[0] - dy, y]
        return result
import collections

class DetectSquares2(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__points = []
        self.__point_counts = collections.defaultdict(int)

    def add(self, point):
        if False:
            while True:
                i = 10
        '\n        :type point: List[int]\n        :rtype: None\n        '
        self.__points.append(point)
        self.__point_counts[tuple(point)] += 1

    def count(self, point):
        if False:
            return 10
        '\n        :type point: List[int]\n        :rtype: int\n        '
        result = 0
        for (x, y) in self.__points:
            if not (point[0] != x and point[1] != y and (abs(point[0] - x) == abs(point[1] - y))):
                continue
            result += self.__point_counts[point[0], y] * self.__point_counts[x, point[1]]
        return result