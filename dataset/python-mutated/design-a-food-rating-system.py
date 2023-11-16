import collections
import itertools
from sortedcontainers import SortedList

class FoodRatings(object):

    def __init__(self, foods, cuisines, ratings):
        if False:
            i = 10
            return i + 15
        '\n        :type foods: List[str]\n        :type cuisines: List[str]\n        :type ratings: List[int]\n        '
        self.__food_to_cuisine = {}
        self.__food_to_rating = {}
        self.__cusine_to_rating_foods = collections.defaultdict(SortedList)
        for (food, cuisine, rating) in itertools.izip(foods, cuisines, ratings):
            self.__food_to_cuisine[food] = cuisine
            self.__food_to_rating[food] = rating
            self.__cusine_to_rating_foods[cuisine].add((-rating, food))

    def changeRating(self, food, newRating):
        if False:
            i = 10
            return i + 15
        '\n        :type food: str\n        :type newRating: int\n        :rtype: None\n        '
        old_rating = self.__food_to_rating[food]
        cuisine = self.__food_to_cuisine[food]
        self.__cusine_to_rating_foods[cuisine].remove((-old_rating, food))
        self.__food_to_rating[food] = newRating
        self.__cusine_to_rating_foods[cuisine].add((-newRating, food))

    def highestRated(self, cuisine):
        if False:
            i = 10
            return i + 15
        '\n        :type cuisine: str\n        :rtype: str\n        '
        return self.__cusine_to_rating_foods[cuisine][0][1]