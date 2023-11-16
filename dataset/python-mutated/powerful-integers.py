import math

class Solution(object):

    def powerfulIntegers(self, x, y, bound):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :type y: int\n        :type bound: int\n        :rtype: List[int]\n        '
        result = set()
        log_x = int(math.floor(math.log(bound) / math.log(x))) + 1 if x != 1 else 1
        log_y = int(math.floor(math.log(bound) / math.log(y))) + 1 if y != 1 else 1
        pow_x = 1
        for i in xrange(log_x):
            pow_y = 1
            for j in xrange(log_y):
                val = pow_x + pow_y
                if val <= bound:
                    result.add(val)
                pow_y *= y
            pow_x *= x
        return list(result)