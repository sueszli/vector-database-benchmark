class Solution(object):

    def getMaximumConsecutive(self, coins):
        if False:
            return 10
        '\n        :type coins: List[int]\n        :rtype: int\n        '
        coins.sort()
        result = 1
        for c in coins:
            if c > result:
                break
            result += c
        return result