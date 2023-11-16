class Solution(object):

    def maximumBags(self, capacity, rocks, additionalRocks):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type capacity: List[int]\n        :type rocks: List[int]\n        :type additionalRocks: int\n        :rtype: int\n        '
        for i in xrange(len(capacity)):
            capacity[i] -= rocks[i]
        capacity.sort()
        for (i, c) in enumerate(capacity):
            if c > additionalRocks:
                return i
            additionalRocks -= c
        return len(capacity)