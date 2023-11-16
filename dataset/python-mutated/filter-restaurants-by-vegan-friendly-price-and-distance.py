class Solution(object):

    def filterRestaurants(self, restaurants, veganFriendly, maxPrice, maxDistance):
        if False:
            print('Hello World!')
        '\n        :type restaurants: List[List[int]]\n        :type veganFriendly: int\n        :type maxPrice: int\n        :type maxDistance: int\n        :rtype: List[int]\n        '
        (result, lookup) = ([], {})
        for (j, (i, _, v, p, d)) in enumerate(restaurants):
            if v >= veganFriendly and p <= maxPrice and (d <= maxDistance):
                lookup[i] = j
                result.append(i)
        result.sort(key=lambda i: (-restaurants[lookup[i]][1], -restaurants[lookup[i]][0]))
        return result