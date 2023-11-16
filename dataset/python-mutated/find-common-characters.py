import collections

class Solution(object):

    def commonChars(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[str]\n        :rtype: List[str]\n        '
        result = collections.Counter(A[0])
        for a in A:
            result &= collections.Counter(a)
        return list(result.elements())