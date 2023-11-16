class Solution(object):

    def minKnightMoves(self, x, y):
        if False:
            return 10
        '\n        :type x: int\n        :type y: int\n        :rtype: int\n        '
        (x, y) = (abs(x), abs(y))
        if x < y:
            (x, y) = (y, x)
        lookup = {(1, 0): 3, (2, 2): 4}
        if (x, y) in lookup:
            return lookup[x, y]
        k = x - y
        if y > k:
            return k - 2 * ((k - y) // 3)
        return k - 2 * ((k - y) // 4)

class Solution2(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__lookup = {(0, 0): 0, (1, 1): 2, (1, 0): 3}

    def minKnightMoves(self, x, y):
        if False:
            print('Hello World!')
        '\n        :type x: int\n        :type y: int\n        :rtype: int\n        '

        def dp(x, y):
            if False:
                print('Hello World!')
            (x, y) = (abs(x), abs(y))
            if x < y:
                (x, y) = (y, x)
            if (x, y) not in self.__lookup:
                self.__lookup[x, y] = min(dp(x - 1, y - 2), dp(x - 2, y - 1)) + 1
            return self.__lookup[x, y]
        return dp(x, y)