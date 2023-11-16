import itertools

class Solution(object):

    def maxCoins(self, piles):
        if False:
            while True:
                i = 10
        '\n        :type piles: List[int]\n        :rtype: int\n        '
        piles.sort()
        return sum(itertools.islice(piles, len(piles) // 3, len(piles), 2))