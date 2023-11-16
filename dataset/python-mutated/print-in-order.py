import threading

class Foo(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__cv = threading.Condition()
        self.__has_first = False
        self.__has_second = False

    def first(self, printFirst):
        if False:
            print('Hello World!')
        '\n        :type printFirst: method\n        :rtype: void\n        '
        with self.__cv:
            printFirst()
            self.__has_first = True
            self.__cv.notifyAll()

    def second(self, printSecond):
        if False:
            print('Hello World!')
        '\n        :type printSecond: method\n        :rtype: void\n        '
        with self.__cv:
            while not self.__has_first:
                self.__cv.wait()
            printSecond()
            self.__has_second = True
            self.__cv.notifyAll()

    def third(self, printThird):
        if False:
            i = 10
            return i + 15
        '\n        :type printThird: method\n        :rtype: void\n        '
        with self.__cv:
            while not self.__has_second:
                self.__cv.wait()
            printThird()
            self.__cv.notifyAll()