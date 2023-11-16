import threading

class FizzBuzz(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.__n = n
        self.__curr = 0
        self.__cv = threading.Condition()

    def fizz(self, printFizz):
        if False:
            i = 10
            return i + 15
        '\n        :type printFizz: method\n        :rtype: void\n        '
        for i in xrange(1, self.__n + 1):
            with self.__cv:
                while self.__curr % 4 != 0:
                    self.__cv.wait()
                self.__curr += 1
                if i % 3 == 0 and i % 5 != 0:
                    printFizz()
                self.__cv.notify_all()

    def buzz(self, printBuzz):
        if False:
            while True:
                i = 10
        '\n        :type printBuzz: method\n        :rtype: void\n        '
        for i in xrange(1, self.__n + 1):
            with self.__cv:
                while self.__curr % 4 != 1:
                    self.__cv.wait()
                self.__curr += 1
                if i % 3 != 0 and i % 5 == 0:
                    printBuzz()
                self.__cv.notify_all()

    def fizzbuzz(self, printFizzBuzz):
        if False:
            return 10
        '\n        :type printFizzBuzz: method\n        :rtype: void\n        '
        for i in xrange(1, self.__n + 1):
            with self.__cv:
                while self.__curr % 4 != 2:
                    self.__cv.wait()
                self.__curr += 1
                if i % 3 == 0 and i % 5 == 0:
                    printFizzBuzz()
                self.__cv.notify_all()

    def number(self, printNumber):
        if False:
            print('Hello World!')
        '\n        :type printNumber: method\n        :rtype: void\n        '
        for i in xrange(1, self.__n + 1):
            with self.__cv:
                while self.__curr % 4 != 3:
                    self.__cv.wait()
                self.__curr += 1
                if i % 3 != 0 and i % 5 != 0:
                    printNumber(i)
                self.__cv.notify_all()