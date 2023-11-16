class Solution(object):

    def numWaterBottles(self, numBottles, numExchange):
        if False:
            while True:
                i = 10
        '\n        :type numBottles: int\n        :type numExchange: int\n        :rtype: int\n        '
        result = numBottles
        while numBottles >= numExchange:
            (numBottles, remainder) = divmod(numBottles, numExchange)
            result += numBottles
            numBottles += remainder
        return result