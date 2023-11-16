class StockSpanner(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__s = []

    def next(self, price):
        if False:
            print('Hello World!')
        '\n        :type price: int\n        :rtype: int\n        '
        result = 1
        while self.__s and self.__s[-1][0] <= price:
            result += self.__s.pop()[1]
        self.__s.append([price, result])
        return result