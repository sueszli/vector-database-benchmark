class ATM(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__vals = [20, 50, 100, 200, 500]
        self.__cnt = [0] * len(self.__vals)

    def deposit(self, banknotesCount):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type banknotesCount: List[int]\n        :rtype: None\n        '
        for (i, x) in enumerate(banknotesCount):
            self.__cnt[i] += x

    def withdraw(self, amount):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type amount: int\n        :rtype: List[int]\n        '
        result = [0] * len(self.__cnt)
        for i in reversed(xrange(len(self.__vals))):
            result[i] = min(amount // self.__vals[i], self.__cnt[i])
            amount -= result[i] * self.__vals[i]
        if amount:
            return [-1]
        for (i, c) in enumerate(result):
            self.__cnt[i] -= c
        return result