class Solution(object):

    def maxArea(self, h, w, horizontalCuts, verticalCuts):
        if False:
            print('Hello World!')
        '\n        :type h: int\n        :type w: int\n        :type horizontalCuts: List[int]\n        :type verticalCuts: List[int]\n        :rtype: int\n        '

        def max_len(l, cuts):
            if False:
                return 10
            cuts.sort()
            l = max(cuts[0] - 0, l - cuts[-1])
            for i in xrange(1, len(cuts)):
                l = max(l, cuts[i] - cuts[i - 1])
            return l
        MOD = 10 ** 9 + 7
        return max_len(h, horizontalCuts) * max_len(w, verticalCuts) % MOD