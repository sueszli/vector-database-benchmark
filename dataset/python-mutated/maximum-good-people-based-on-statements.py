class Solution(object):

    def maximumGood(self, statements):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type statements: List[List[int]]\n        :rtype: int\n        '

        def check(mask):
            if False:
                i = 10
                return i + 15
            return all((mask >> j & 1 == statements[i][j] for i in xrange(len(statements)) if mask >> i & 1 for j in xrange(len(statements[i])) if statements[i][j] != 2))

        def popcount(x):
            if False:
                while True:
                    i = 10
            result = 0
            while x:
                x &= x - 1
                result += 1
            return result
        result = 0
        for mask in xrange(1 << len(statements)):
            if check(mask):
                result = max(result, popcount(mask))
        return result