import collections

class BIT(object):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.__bit = [0] * n

    def add(self, i, val):
        if False:
            print('Hello World!')
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def sum(self, i):
        if False:
            while True:
                i = 10
        result = 0
        while i > 0:
            result += self.__bit[i]
            i -= i & -i
        return result

class Solution(object):

    def minInteger(self, num, k):
        if False:
            return 10
        '\n        :type num: str\n        :type k: int\n        :rtype: str\n        '
        lookup = collections.defaultdict(list)
        bit = BIT(len(num) + 1)
        for i in reversed(xrange(len(num))):
            bit.add(i + 1, 1)
            lookup[int(num[i])].append(i + 1)
        result = []
        for _ in xrange(len(num)):
            for d in xrange(10):
                if lookup[d] and bit.sum(lookup[d][-1] - 1) <= k:
                    k -= bit.sum(lookup[d][-1] - 1)
                    bit.add(lookup[d].pop(), -1)
                    result.append(d)
                    break
        return ''.join(map(str, result))