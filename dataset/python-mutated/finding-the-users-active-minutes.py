import collections

class Solution(object):

    def findingUsersActiveMinutes(self, logs, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type logs: List[List[int]]\n        :type k: int\n        :rtype: List[int]\n        '
        lookup = collections.defaultdict(set)
        for (u, t) in logs:
            lookup[u].add(t)
        result = [0] * k
        for (_, ts) in lookup.iteritems():
            result[len(ts) - 1] += 1
        return result