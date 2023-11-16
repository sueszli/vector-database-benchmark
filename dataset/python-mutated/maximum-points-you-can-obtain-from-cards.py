class Solution(object):

    def maxScore(self, cardPoints, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type cardPoints: List[int]\n        :type k: int\n        :rtype: int\n        '
        (result, total, curr, left) = (float('inf'), 0, 0, 0)
        for (right, point) in enumerate(cardPoints):
            total += point
            curr += point
            if right - left + 1 > len(cardPoints) - k:
                curr -= cardPoints[left]
                left += 1
            if right - left + 1 == len(cardPoints) - k:
                result = min(result, curr)
        return total - result