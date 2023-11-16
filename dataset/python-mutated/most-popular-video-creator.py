import collections
import itertools

class Solution(object):

    def mostPopularCreator(self, creators, ids, views):
        if False:
            print('Hello World!')
        '\n        :type creators: List[str]\n        :type ids: List[str]\n        :type views: List[int]\n        :rtype: List[List[str]]\n        '
        cnt = collections.Counter()
        lookup = collections.defaultdict(lambda : (float('inf'), float('inf')))
        for (c, i, v) in itertools.izip(creators, ids, views):
            cnt[c] += v
            lookup[c] = min(lookup[c], (-v, i))
        mx = max(cnt.itervalues())
        return [[k, lookup[k][1]] for (k, v) in cnt.iteritems() if v == mx]