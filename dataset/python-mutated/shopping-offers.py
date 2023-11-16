class Solution(object):

    def shoppingOffers(self, price, special, needs):
        if False:
            while True:
                i = 10
        '\n        :type price: List[int]\n        :type special: List[List[int]]\n        :type needs: List[int]\n        :rtype: int\n        '

        def shoppingOffersHelper(price, special, needs, i):
            if False:
                return 10
            if i == len(special):
                return sum(map(lambda x, y: x * y, price, needs))
            result = shoppingOffersHelper(price, special, needs, i + 1)
            for j in xrange(len(needs)):
                needs[j] -= special[i][j]
            if all((need >= 0 for need in needs)):
                result = min(result, special[i][-1] + shoppingOffersHelper(price, special, needs, i))
            for j in xrange(len(needs)):
                needs[j] += special[i][j]
            return result
        return shoppingOffersHelper(price, special, needs, 0)