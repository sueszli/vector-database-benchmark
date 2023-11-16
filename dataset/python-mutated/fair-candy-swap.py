class Solution(object):

    def fairCandySwap(self, A, B):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type B: List[int]\n        :rtype: List[int]\n        '
        diff = (sum(A) - sum(B)) // 2
        setA = set(A)
        for b in set(B):
            if diff + b in setA:
                return [diff + b, b]
        return []