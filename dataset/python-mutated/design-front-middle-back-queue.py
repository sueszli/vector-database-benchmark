import collections

class FrontMiddleBackQueue(object):

    def __init__(self):
        if False:
            print('Hello World!')
        (self.__left, self.__right) = (collections.deque(), collections.deque())

    def pushFront(self, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type val: int\n        :rtype: None\n        '
        self.__left.appendleft(val)
        self.__balance()

    def pushMiddle(self, val):
        if False:
            while True:
                i = 10
        '\n        :type val: int\n        :rtype: None\n        '
        if len(self.__left) > len(self.__right):
            self.__right.appendleft(self.__left.pop())
        self.__left.append(val)

    def pushBack(self, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type val: int\n        :rtype: None\n        '
        self.__right.append(val)
        self.__balance()

    def popFront(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        val = (self.__left or collections.deque([-1])).popleft()
        self.__balance()
        return val

    def popMiddle(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: int\n        '
        val = (self.__left or [-1]).pop()
        self.__balance()
        return val

    def popBack(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        val = (self.__right or self.__left or [-1]).pop()
        self.__balance()
        return val

    def __balance(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.__left) > len(self.__right) + 1:
            self.__right.appendleft(self.__left.pop())
        elif len(self.__left) < len(self.__right):
            self.__left.append(self.__right.popleft())