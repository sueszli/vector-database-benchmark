import collections

class Solution(object):

    def hasGroupsSizeX(self, deck):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type deck: List[int]\n        :rtype: bool\n        '

        def gcd(a, b):
            if False:
                print('Hello World!')
            while b:
                (a, b) = (b, a % b)
            return a
        vals = collections.Counter(deck).values()
        return reduce(gcd, vals) >= 2