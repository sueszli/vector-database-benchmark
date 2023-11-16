class Node(object):

    def __init__(self, start, end):
        if False:
            return 10
        self.__start = start
        self.__end = end
        self.__left = None
        self.__right = None

    def insert(self, node):
        if False:
            i = 10
            return i + 15
        if node.__start >= self.__end:
            if not self.__right:
                self.__right = node
                return True
            return self.__right.insert(node)
        elif node.__end <= self.__start:
            if not self.__left:
                self.__left = node
                return True
            return self.__left.insert(node)
        else:
            return False

class MyCalendar(object):

    def __init__(self):
        if False:
            return 10
        self.__root = None

    def book(self, start, end):
        if False:
            print('Hello World!')
        '\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '
        if self.__root is None:
            self.__root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))

class MyCalendar2(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__calendar = []

    def book(self, start, end):
        if False:
            i = 10
            return i + 15
        '\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '
        for (i, j) in self.__calendar:
            if start < j and end > i:
                return False
        self.__calendar.append((start, end))
        return True