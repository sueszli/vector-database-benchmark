import collections

class Solution(object):

    def largestMerge(self, word1, word2):
        if False:
            return 10
        '\n        :type word1: str\n        :type word2: str\n        :rtype: str\n        '
        q1 = collections.deque(word1)
        q2 = collections.deque(word2)
        result = []
        while q1 or q2:
            if q1 > q2:
                result.append(q1.popleft())
            else:
                result.append(q2.popleft())
        return ''.join(result)