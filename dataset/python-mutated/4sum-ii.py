import collections

class Solution(object):

    def fourSumCount(self, A, B, C, D):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type B: List[int]\n        :type C: List[int]\n        :type D: List[int]\n        :rtype: int\n        '
        A_B_sum = collections.Counter((a + b for a in A for b in B))
        return sum((A_B_sum[-c - d] for c in C for d in D))