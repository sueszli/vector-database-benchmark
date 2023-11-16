import threading

class ZeroEvenOdd(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.__n = n
        self.__curr = 0
        self.__cv = threading.Condition()

    def zero(self, printNumber):
        if False:
            i = 10
            return i + 15
        '\n        :type printNumber: method\n        :rtype: void\n        '
        for i in xrange(self.__n):
            with self.__cv:
                while self.__curr % 2 != 0:
                    self.__cv.wait()
                self.__curr += 1
                printNumber(0)
                self.__cv.notifyAll()

    def even(self, printNumber):
        if False:
            print('Hello World!')
        '\n        :type printNumber: method\n        :rtype: void\n        '
        for i in xrange(2, self.__n + 1, 2):
            with self.__cv:
                while self.__curr % 4 != 3:
                    self.__cv.wait()
                self.__curr += 1
                printNumber(i)
                self.__cv.notifyAll()

    def odd(self, printNumber):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type printNumber: method\n        :rtype: void\n        '
        for i in xrange(1, self.__n + 1, 2):
            with self.__cv:
                while self.__curr % 4 != 1:
                    self.__cv.wait()
                self.__curr += 1
                printNumber(i)
                self.__cv.notifyAll()