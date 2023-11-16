import fractions

class Solution(object):

    def simplifiedFractions(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: List[str]\n        '
        lookup = set()
        for b in xrange(1, n + 1):
            for a in xrange(1, b):
                g = fractions.gcd(a, b)
                lookup.add((a // g, b // g))
        return map(lambda x: '{}/{}'.format(*x), lookup)