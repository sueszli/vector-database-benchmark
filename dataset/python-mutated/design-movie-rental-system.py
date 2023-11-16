import collections
from sortedcontainers import SortedList

class MovieRentingSystem(object):

    def __init__(self, n, entries):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type entries: List[List[int]]\n        '
        self.__movie_to_ordered_price_shop = collections.defaultdict(SortedList)
        self.__shop_movie_to_price = {}
        self.__rented_ordered_price_shop_movie = SortedList()
        for (s, m, p) in entries:
            self.__movie_to_ordered_price_shop[m].add((p, s))
            self.__shop_movie_to_price[s, m] = p

    def search(self, movie):
        if False:
            print('Hello World!')
        '\n        :type movie: int\n        :rtype: List[int]\n        '
        return [s for (_, s) in self.__movie_to_ordered_price_shop[movie][:5]]

    def rent(self, shop, movie):
        if False:
            i = 10
            return i + 15
        '\n        :type shop: int\n        :type movie: int\n        :rtype: None\n        '
        price = self.__shop_movie_to_price[shop, movie]
        self.__movie_to_ordered_price_shop[movie].remove((price, shop))
        self.__rented_ordered_price_shop_movie.add((price, shop, movie))

    def drop(self, shop, movie):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type shop: int\n        :type movie: int\n        :rtype: None\n        '
        price = self.__shop_movie_to_price[shop, movie]
        self.__movie_to_ordered_price_shop[movie].add((price, shop))
        self.__rented_ordered_price_shop_movie.remove((price, shop, movie))

    def report(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: List[List[int]]\n        '
        return [[s, m] for (_, s, m) in self.__rented_ordered_price_shop_movie[:5]]