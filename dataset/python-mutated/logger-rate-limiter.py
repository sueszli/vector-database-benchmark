import collections

class Logger(object):

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize your data structure here.\n        '
        self.__dq = collections.deque()
        self.__printed = set()

    def shouldPrintMessage(self, timestamp, message):
        if False:
            return 10
        '\n        Returns true if the message should be printed in the given timestamp, otherwise returns false. The timestamp is in seconds granularity.\n        :type timestamp: int\n        :type message: str\n        :rtype: bool\n        '
        while self.__dq and self.__dq[0][0] <= timestamp - 10:
            self.__printed.remove(self.__dq.popleft()[1])
        if message in self.__printed:
            return False
        self.__dq.append((timestamp, message))
        self.__printed.add(message)
        return True