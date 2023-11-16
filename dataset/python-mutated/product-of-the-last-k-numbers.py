class ProductOfNumbers(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__accu = [1]

    def add(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: None\n        '
        if not num:
            self.__accu = [1]
            return
        self.__accu.append(self.__accu[-1] * num)

    def getProduct(self, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type k: int\n        :rtype: int\n        '
        if len(self.__accu) <= k:
            return 0
        return self.__accu[-1] // self.__accu[-1 - k]