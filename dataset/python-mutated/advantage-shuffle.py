class Solution(object):

    def advantageCount(self, A, B):
        if False:
            while True:
                i = 10
        '\n        :type A: List[int]\n        :type B: List[int]\n        :rtype: List[int]\n        '
        sortedA = sorted(A)
        sortedB = sorted(B)
        candidates = {b: [] for b in B}
        others = []
        j = 0
        for a in sortedA:
            if a > sortedB[j]:
                candidates[sortedB[j]].append(a)
                j += 1
            else:
                others.append(a)
        return [candidates[b].pop() if candidates[b] else others.pop() for b in B]