class Solution(object):

    def minMoves(self, target, maxDoubles):
        if False:
            while True:
                i = 10
        '\n        :type target: int\n        :type maxDoubles: int\n        :rtype: int\n        '
        result = 0
        while target > 1 and maxDoubles:
            result += 1 + target % 2
            target //= 2
            maxDoubles -= 1
        return result + (target - 1)