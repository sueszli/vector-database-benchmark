class Solution(object):

    def maximumBobPoints(self, numArrows, aliceArrows):
        if False:
            print('Hello World!')
        '\n        :type numArrows: int\n        :type aliceArrows: List[int]\n        :rtype: List[int]\n        '

        def check(mask, numArrows):
            if False:
                i = 10
                return i + 15
            score = 0
            cnt = [0] * len(aliceArrows)
            (i, base) = (0, 1)
            for (k, a) in enumerate(aliceArrows):
                if mask & 1:
                    need = a + 1
                    if need > numArrows:
                        return (0, [0] * len(aliceArrows))
                    numArrows -= need
                    cnt[k] = need
                    score += k
                mask >>= 1
            cnt[-1] += numArrows
            return (score, cnt)
        result = [0] * len(aliceArrows)
        best = 0
        for mask in xrange(1, 2 ** len(aliceArrows)):
            (score, cnt) = check(mask, numArrows)
            if score > best:
                best = score
                result = cnt
        return result