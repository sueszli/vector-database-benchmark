import operator
import collections
from functools import reduce

class Solution(object):

    def findTheDifference(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :rtype: str\n        '
        return chr(reduce(operator.xor, map(ord, s), 0) ^ reduce(operator.xor, map(ord, t), 0))

    def findTheDifference2(self, s, t):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type t: str\n        :rtype: str\n        '
        t = list(t)
        s = list(s)
        for i in s:
            t.remove(i)
        return t[0]

    def findTheDifference3(self, s, t):
        if False:
            while True:
                i = 10
        return chr(reduce(operator.xor, map(ord, s + t)))

    def findTheDifference4(self, s, t):
        if False:
            for i in range(10):
                print('nop')
        return list(collections.Counter(t) - collections.Counter(s))[0]

    def findTheDifference5(self, s, t):
        if False:
            print('Hello World!')
        (s, t) = (sorted(s), sorted(t))
        return t[-1] if s == t[:-1] else [x[1] for x in zip(s, t) if x[0] != x[1]][0]