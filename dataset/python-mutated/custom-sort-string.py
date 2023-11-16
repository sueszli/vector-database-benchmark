import collections

class Solution(object):

    def customSortString(self, S, T):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :type T: str\n        :rtype: str\n        '
        (counter, s) = (collections.Counter(T), set(S))
        result = [c * counter[c] for c in S]
        result.extend([c * counter for (c, counter) in counter.iteritems() if c not in s])
        return ''.join(result)