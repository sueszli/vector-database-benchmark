class TextEditor(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__LAST_COUNT = 10
        self.__left = []
        self.__right = []

    def addText(self, text):
        if False:
            print('Hello World!')
        '\n        :type text: str\n        :rtype: None\n        '
        for x in text:
            self.__left.append(x)

    def deleteText(self, k):
        if False:
            while True:
                i = 10
        '\n        :type k: int\n        :rtype: int\n        '
        return self.__move(k, self.__left, None)

    def cursorLeft(self, k):
        if False:
            i = 10
            return i + 15
        '\n        :type k: int\n        :rtype: str\n        '
        self.__move(k, self.__left, self.__right)
        return self.__last_characters()

    def cursorRight(self, k):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :rtype: str\n        '
        self.__move(k, self.__right, self.__left)
        return self.__last_characters()

    def __move(self, k, src, dst):
        if False:
            print('Hello World!')
        cnt = min(k, len(src))
        for _ in xrange(cnt):
            if dst is not None:
                dst.append(src[-1])
            src.pop()
        return cnt

    def __last_characters(self):
        if False:
            for i in range(10):
                print('nop')
        return ''.join(self.__left[-self.__LAST_COUNT:])