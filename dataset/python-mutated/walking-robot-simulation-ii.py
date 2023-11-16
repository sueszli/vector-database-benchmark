class Robot(object):

    def __init__(self, width, height):
        if False:
            while True:
                i = 10
        '\n        :type width: int\n        :type height: int\n        '
        self.__w = width
        self.__h = height
        self.__curr = 0

    def move(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: None\n        '
        self.__curr += num

    def getPos(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: List[int]\n        '
        n = self.__curr % (2 * (self.__w - 1 + (self.__h - 1)))
        if n < self.__w:
            return [n, 0]
        n -= self.__w - 1
        if n < self.__h:
            return [self.__w - 1, n]
        n -= self.__h - 1
        if n < self.__w:
            return [self.__w - 1 - n, self.__h - 1]
        n -= self.__w - 1
        return [0, self.__h - 1 - n]

    def getDir(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: str\n        '
        n = self.__curr % (2 * (self.__w - 1 + (self.__h - 1)))
        if n < self.__w:
            return 'South' if n == 0 and self.__curr else 'East'
        n -= self.__w - 1
        if n < self.__h:
            return 'North'
        n -= self.__h - 1
        if n < self.__w:
            return 'West'
        n -= self.__w - 1
        return 'South'

class Robot2(object):

    def __init__(self, width, height):
        if False:
            i = 10
            return i + 15
        '\n        :type width: int\n        :type height: int\n        '
        self.__w = width
        self.__h = height
        self.__curr = 0

    def move(self, num):
        if False:
            print('Hello World!')
        '\n        :type num: int\n        :rtype: None\n        '
        self.__curr += num

    def getPos(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: List[int]\n        '
        return self.__getPosDir()[0]

    def getDir(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: str\n        '
        return self.__getPosDir()[1]

    def __getPosDir(self):
        if False:
            i = 10
            return i + 15
        n = self.__curr % (2 * (self.__w - 1 + (self.__h - 1)))
        if n < self.__w:
            return [[n, 0], 'South' if n == 0 and self.__curr else 'East']
        n -= self.__w - 1
        if n < self.__h:
            return [[self.__w - 1, n], 'North']
        n -= self.__h - 1
        if n < self.__w:
            return [[self.__w - 1 - n, self.__h - 1], 'West']
        n -= self.__w - 1
        return [[0, self.__h - 1 - n], 'South']