class MyCircularDeque(object):

    def __init__(self, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here. Set the size of the deque to be k.\n        :type k: int\n        '
        self.__start = 0
        self.__size = 0
        self.__buffer = [0] * k

    def insertFront(self, value):
        if False:
            while True:
                i = 10
        '\n        Adds an item at the front of Deque. Return true if the operation is successful.\n        :type value: int\n        :rtype: bool\n        '
        if self.isFull():
            return False
        self.__start = (self.__start - 1) % len(self.__buffer)
        self.__buffer[self.__start] = value
        self.__size += 1
        return True

    def insertLast(self, value):
        if False:
            return 10
        '\n        Adds an item at the rear of Deque. Return true if the operation is successful.\n        :type value: int\n        :rtype: bool\n        '
        if self.isFull():
            return False
        self.__buffer[(self.__start + self.__size) % len(self.__buffer)] = value
        self.__size += 1
        return True

    def deleteFront(self):
        if False:
            while True:
                i = 10
        '\n        Deletes an item from the front of Deque. Return true if the operation is successful.\n        :rtype: bool\n        '
        if self.isEmpty():
            return False
        self.__start = (self.__start + 1) % len(self.__buffer)
        self.__size -= 1
        return True

    def deleteLast(self):
        if False:
            while True:
                i = 10
        '\n        Deletes an item from the rear of Deque. Return true if the operation is successful.\n        :rtype: bool\n        '
        if self.isEmpty():
            return False
        self.__size -= 1
        return True

    def getFront(self):
        if False:
            while True:
                i = 10
        '\n        Get the front item from the deque.\n        :rtype: int\n        '
        return -1 if self.isEmpty() else self.__buffer[self.__start]

    def getRear(self):
        if False:
            while True:
                i = 10
        '\n        Get the last item from the deque.\n        :rtype: int\n        '
        return -1 if self.isEmpty() else self.__buffer[(self.__start + self.__size - 1) % len(self.__buffer)]

    def isEmpty(self):
        if False:
            while True:
                i = 10
        '\n        Checks whether the circular deque is empty or not.\n        :rtype: bool\n        '
        return self.__size == 0

    def isFull(self):
        if False:
            return 10
        '\n        Checks whether the circular deque is full or not.\n        :rtype: bool\n        '
        return self.__size == len(self.__buffer)