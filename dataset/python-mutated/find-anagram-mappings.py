import collections

class Solution(object):

    def anagramMappings(self, A, B):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type B: List[int]\n        :rtype: List[int]\n        '
        lookup = collections.defaultdict(collections.deque)
        for (i, n) in enumerate(B):
            lookup[n].append(i)
        result = []
        for n in A:
            result.append(lookup[n].popleft())
        return result