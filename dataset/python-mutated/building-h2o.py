import threading

class H2O(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__l = threading.Lock()
        self.__nH = 0
        self.__nO = 0
        self.__releaseHydrogen = None
        self.__releaseOxygen = None

    def hydrogen(self, releaseHydrogen):
        if False:
            i = 10
            return i + 15
        with self.__l:
            self.__releaseHydrogen = releaseHydrogen
            self.__nH += 1
            self.__output()

    def oxygen(self, releaseOxygen):
        if False:
            print('Hello World!')
        with self.__l:
            self.__releaseOxygen = releaseOxygen
            self.__nO += 1
            self.__output()

    def __output(self):
        if False:
            i = 10
            return i + 15
        while self.__nH >= 2 and self.__nO >= 1:
            self.__nH -= 2
            self.__nO -= 1
            self.__releaseHydrogen()
            self.__releaseHydrogen()
            self.__releaseOxygen()

class H2O2(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__nH = 0
        self.__nO = 0
        self.__cv = threading.Condition()

    def hydrogen(self, releaseHydrogen):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type releaseHydrogen: method\n        :rtype: void\n        '
        with self.__cv:
            while self.__nH + 1 - 2 * self.__nO > 2:
                self.__cv.wait()
            self.__nH += 1
            releaseHydrogen()
            self.__cv.notifyAll()

    def oxygen(self, releaseOxygen):
        if False:
            i = 10
            return i + 15
        '\n        :type releaseOxygen: method\n        :rtype: void\n        '
        with self.__cv:
            while 2 * (self.__nO + 1) - self.__nH > 2:
                self.__cv.wait()
            self.__nO += 1
            releaseOxygen()
            self.__cv.notifyAll()