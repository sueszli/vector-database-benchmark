class BIT(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.__bit = [0] * n

    def add(self, i, val):
        if False:
            i = 10
            return i + 15
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def sum(self, i):
        if False:
            return 10
        result = 0
        while i > 0:
            result += self.__bit[i]
            i -= i & -i
        return result

class Solution(object):

    def processQueries(self, queries, m):
        if False:
            print('Hello World!')
        '\n        :type queries: List[int]\n        :type m: int\n        :rtype: List[int]\n        '
        bit = BIT(2 * m + 1)
        lookup = {}
        for i in xrange(1, m + 1):
            bit.add(m + i, 1)
            lookup[i] = m + i
        (result, curr) = ([], m)
        for q in queries:
            i = lookup.pop(q)
            result.append(bit.sum(i - 1))
            bit.add(i, -1)
            lookup[q] = curr
            bit.add(curr, 1)
            curr -= 1
        return result