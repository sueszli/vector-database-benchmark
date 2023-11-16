MOD = 10 ** 9 + 7

class Fancy(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__arr = []
        self.__ops = [[1, 0]]

    def append(self, val):
        if False:
            return 10
        '\n        :type val: int\n        :rtype: None\n        '
        self.__arr.append(val)
        self.__ops.append(self.__ops[-1][:])

    def addAll(self, inc):
        if False:
            return 10
        '\n        :type inc: int\n        :rtype: None\n        '
        self.__ops[-1][1] = (self.__ops[-1][1] + inc) % MOD

    def multAll(self, m):
        if False:
            while True:
                i = 10
        '\n        :type m: int\n        :rtype: None\n        '
        self.__ops[-1] = [self.__ops[-1][0] * m % MOD, self.__ops[-1][1] * m % MOD]

    def getIndex(self, idx):
        if False:
            print('Hello World!')
        '\n        :type idx: int\n        :rtype: int\n        '
        if idx >= len(self.__arr):
            return -1
        (a1, b1) = self.__ops[idx]
        (a2, b2) = self.__ops[-1]
        a = a2 * pow(a1, MOD - 2, MOD) % MOD
        b = (b2 - b1 * a) % MOD
        return (self.__arr[idx] * a + b) % MOD

class Fancy2(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__arr = []
        self.__op = [1, 0]

    def append(self, val):
        if False:
            i = 10
            return i + 15
        '\n        :type val: int\n        :rtype: None\n        '
        self.__arr.append((val - self.__op[1]) * pow(self.__op[0], MOD - 2, MOD) % MOD)

    def addAll(self, inc):
        if False:
            print('Hello World!')
        '\n        :type inc: int\n        :rtype: None\n        '
        self.__op[1] = (self.__op[1] + inc) % MOD

    def multAll(self, m):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type m: int\n        :rtype: None\n        '
        self.__op = [self.__op[0] * m % MOD, self.__op[1] * m % MOD]

    def getIndex(self, idx):
        if False:
            i = 10
            return i + 15
        '\n        :type idx: int\n        :rtype: int\n        '
        if idx >= len(self.__arr):
            return -1
        (a, b) = self.__op
        return (self.__arr[idx] * a + b) % MOD