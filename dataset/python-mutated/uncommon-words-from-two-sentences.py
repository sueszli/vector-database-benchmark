import collections

class Solution(object):

    def uncommonFromSentences(self, A, B):
        if False:
            i = 10
            return i + 15
        '\n        :type A: str\n        :type B: str\n        :rtype: List[str]\n        '
        count = collections.Counter(A.split())
        count += collections.Counter(B.split())
        return [word for word in count if count[word] == 1]