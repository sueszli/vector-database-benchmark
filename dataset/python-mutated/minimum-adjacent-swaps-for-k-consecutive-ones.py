class Solution(object):

    def minMoves(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def moves(i, j):
            if False:
                i = 10
                return i + 15
            return prefix[j + 1] - prefix[i]
        idxs = [i for (i, x) in enumerate(nums) if x]
        prefix = [0] * (len(idxs) + 1)
        for i in xrange(len(idxs)):
            prefix[i + 1] = prefix[i] + idxs[i]
        result = float('inf')
        for i in xrange(len(idxs) - k + 1):
            result = min(result, -moves(i, i + k // 2 - 1) + moves(i + (k + 1) // 2, i + k - 1))
        result -= k // 2 * ((k + 1) // 2)
        return result