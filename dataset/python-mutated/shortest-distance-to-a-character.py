import itertools

class Solution(object):

    def shortestToChar(self, S, C):
        if False:
            return 10
        '\n        :type S: str\n        :type C: str\n        :rtype: List[int]\n        '
        result = [len(S)] * len(S)
        prev = -len(S)
        for i in itertools.chain(xrange(len(S)), reversed(xrange(len(S)))):
            if S[i] == C:
                prev = i
            result[i] = min(result[i], abs(i - prev))
        return result