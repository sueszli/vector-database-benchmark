import re

class StringIterator(object):

    def __init__(self, compressedString):
        if False:
            while True:
                i = 10
        '\n        :type compressedString: str\n        '
        self.__result = re.findall('([a-zA-Z])(\\d+)', compressedString)
        (self.__index, self.__num, self.__ch) = (0, 0, ' ')

    def next(self):
        if False:
            print('Hello World!')
        '\n        :rtype: str\n        '
        if not self.hasNext():
            return ' '
        if self.__num == 0:
            self.__ch = self.__result[self.__index][0]
            self.__num = int(self.__result[self.__index][1])
            self.__index += 1
        self.__num -= 1
        return self.__ch

    def hasNext(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: bool\n        '
        return self.__index != len(self.__result) or self.__num != 0