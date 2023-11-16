import collections
import itertools

class Solution(object):

    def mostVisitedPattern(self, username, timestamp, website):
        if False:
            while True:
                i = 10
        '\n        :type username: List[str]\n        :type timestamp: List[int]\n        :type website: List[str]\n        :rtype: List[str]\n        '
        lookup = collections.defaultdict(list)
        A = zip(timestamp, username, website)
        A.sort()
        for (t, u, w) in A:
            lookup[u].append(w)
        count = sum([collections.Counter(set(itertools.combinations(lookup[u], 3))) for u in lookup], collections.Counter())
        return list(min(count, key=lambda x: (-count[x], x)))