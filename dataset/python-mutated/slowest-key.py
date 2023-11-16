import collections

class Solution(object):

    def slowestKey(self, releaseTimes, keysPressed):
        if False:
            return 10
        '\n        :type releaseTimes: List[int]\n        :type keysPressed: str\n        :rtype: str\n        '
        (result, lookup) = ('a', collections.Counter())
        for (i, c) in enumerate(keysPressed):
            lookup[c] = max(lookup[c], releaseTimes[i] - (releaseTimes[i - 1] if i > 0 else 0))
            if lookup[c] > lookup[result] or (lookup[c] == lookup[result] and c > result):
                result = c
        return result