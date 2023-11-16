import operator

class Solution(object):

    def nimGame(self, piles):
        if False:
            print('Hello World!')
        '\n        :type piles: List[int]\n        :rtype: bool\n        '
        return reduce(operator.xor, piles, 0)