MAX_R = 10 ** 4
LOOKUP = [-1] * MAX_R

class Solution(object):

    def countSymmetricIntegers(self, low, high):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type low: int\n        :type high: int\n        :rtype: int\n        '

        def check(x):
            if False:
                while True:
                    i = 10
            if LOOKUP[x - 1] == -1:
                digits = map(int, str(x))
                if len(digits) % 2:
                    LOOKUP[x - 1] = 0
                else:
                    LOOKUP[x - 1] = int(sum((digits[i] for i in xrange(len(digits) // 2))) == sum((digits[i] for i in xrange(len(digits) // 2, len(digits)))))
            return LOOKUP[x - 1]
        return sum((check(x) for x in xrange(low, high + 1)))