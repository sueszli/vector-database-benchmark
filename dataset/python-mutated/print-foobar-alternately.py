import threading

class FooBar(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.__n = n
        self.__curr = False
        self.__cv = threading.Condition()

    def foo(self, printFoo):
        if False:
            i = 10
            return i + 15
        '\n        :type printFoo: method\n        :rtype: void\n        '
        for i in xrange(self.__n):
            with self.__cv:
                while self.__curr != False:
                    self.__cv.wait()
                self.__curr = not self.__curr
                printFoo()
                self.__cv.notify()

    def bar(self, printBar):
        if False:
            print('Hello World!')
        '\n        :type printBar: method\n        :rtype: void\n        '
        for i in xrange(self.__n):
            with self.__cv:
                while self.__curr != True:
                    self.__cv.wait()
                self.__curr = not self.__curr
                printBar()
                self.__cv.notify()