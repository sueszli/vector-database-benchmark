import collections

class Solution(object):

    def getDistances(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '
        lookup = collections.defaultdict(list)
        for (i, x) in enumerate(arr):
            lookup[x].append(i)
        result = [0] * len(arr)
        for idxs in lookup.itervalues():
            prefix = [0]
            for i in idxs:
                prefix.append(prefix[-1] + i)
            for (i, idx) in enumerate(idxs):
                result[idx] = idx * (i + 1) - prefix[i + 1] + (prefix[len(idxs)] - prefix[i] - idx * (len(idxs) - i))
        return result