import collections

class Solution(object):

    def wordSubsets(self, A, B):
        if False:
            while True:
                i = 10
        '\n        :type A: List[str]\n        :type B: List[str]\n        :rtype: List[str]\n        '
        count = collections.Counter()
        for b in B:
            for (c, n) in collections.Counter(b).items():
                count[c] = max(count[c], n)
        result = []
        for a in A:
            count = collections.Counter(a)
            if all((count[c] >= count[c] for c in count)):
                result.append(a)
        return result