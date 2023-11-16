class Solution(object):

    def countTriples(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        lookup = set()
        for i in xrange(1, n + 1):
            lookup.add(i ** 2)
        result = 0
        for i in xrange(1, n + 1):
            for j in xrange(1, n + 1):
                result += int(i ** 2 + j ** 2 in lookup)
        return result