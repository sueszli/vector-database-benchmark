class Solution(object):

    def shortestSequence(self, rolls, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type rolls: List[int]\n        :type k: int\n        :rtype: int\n        '
        l = 0
        lookup = set()
        for x in rolls:
            lookup.add(x)
            if len(lookup) != k:
                continue
            lookup.clear()
            l += 1
        return l + 1