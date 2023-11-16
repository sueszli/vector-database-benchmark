from collections import defaultdict

class Solution(object):

    def firstUniqChar(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        lookup = defaultdict(int)
        candidtates = set()
        for (i, c) in enumerate(s):
            if lookup[c]:
                candidtates.discard(lookup[c])
            else:
                lookup[c] = i + 1
                candidtates.add(i + 1)
        return min(candidtates) - 1 if candidtates else -1