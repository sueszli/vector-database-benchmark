class MyCircularQueue(object):

    def __init__(self, k):
        if False:
            while True:
                i = 10
        '\n        Initialize your data structure here. Set the size of the queue to be k.\n        :type k: int\n        '
        self.__start = 0
        self.__size = 0
        self.__buffer = [0] * k

    def enQueue(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Insert an element into the circular queue. Return true if the operation is successful.\n        :type value: int\n        :rtype: bool\n        '
        if self.isFull():
            return False
        self.__buffer[(self.__start + self.__size) % len(self.__buffer)] = value
        self.__size += 1
        return True

    def deQueue(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete an element from the circular queue. Return true if the operation is successful.\n        :rtype: bool\n        '
        if self.isEmpty():
            return False
        self.__start = (self.__start + 1) % len(self.__buffer)
        self.__size -= 1
        return True

    def Front(self):
        if False:
            return 10
        '\n        Get the front item from the queue.\n        :rtype: int\n        '
        return -1 if self.isEmpty() else self.__buffer[self.__start]

    def Rear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the last item from the queue.\n        :rtype: int\n        '
        return -1 if self.isEmpty() else self.__buffer[(self.__start + self.__size - 1) % len(self.__buffer)]

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        '\n        Checks whether the circular queue is empty or not.\n        :rtype: bool\n        '
        return self.__size == 0

    def isFull(self):
        if False:
            i = 10
            return i + 15
        '\n        Checks whether the circular queue is full or not.\n        :rtype: bool\n        '
        return self.__size == len(self.__buffer)