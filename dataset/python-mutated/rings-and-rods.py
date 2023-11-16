import collections

class Solution(object):

    def countPoints(self, rings):
        if False:
            while True:
                i = 10
        '\n        :type rings: str\n        :rtype: int\n        '
        bits = {'R': 1, 'G': 2, 'B': 4}
        rods = collections.defaultdict(int)
        for i in xrange(0, len(rings), 2):
            rods[int(rings[i + 1])] |= bits[rings[i]]
        return sum((x == 7 for x in rods.itervalues()))