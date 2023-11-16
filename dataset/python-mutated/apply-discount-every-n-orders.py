class Cashier(object):

    def __init__(self, n, discount, products, prices):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type discount: int\n        :type products: List[int]\n        :type prices: List[int]\n        '
        self.__n = n
        self.__discount = discount
        self.__curr = 0
        self.__lookup = {p: prices[i] for (i, p) in enumerate(products)}

    def getBill(self, product, amount):
        if False:
            return 10
        '\n        :type product: List[int]\n        :type amount: List[int]\n        :rtype: float\n        '
        self.__curr = (self.__curr + 1) % self.__n
        result = 0.0
        for (i, p) in enumerate(product):
            result += self.__lookup[p] * amount[i]
        return result * (1.0 - self.__discount / 100.0 if self.__curr == 0 else 1.0)