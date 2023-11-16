import collections

class Solution(object):

    def largestUniqueNumber(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[int]\n        :rtype: int\n        '
        A.append(-1)
        return max((k for (k, v) in collections.Counter(A).items() if v == 1))