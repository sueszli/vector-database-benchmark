import itertools

class Solution(object):

    def ambiguousCoordinates(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: List[str]\n        '

        def make(S, i, n):
            if False:
                return 10
            for d in xrange(1, n + 1):
                left = S[i:i + d]
                right = S[i + d:i + n]
                if (not left.startswith('0') or left == '0') and (not right.endswith('0')):
                    yield ''.join([left, '.' if right else '', right])
        return ['({}, {})'.format(*cand) for i in xrange(1, len(S) - 2) for cand in itertools.product(make(S, 1, i), make(S, i + 1, len(S) - 2 - i))]