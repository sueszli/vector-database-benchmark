class Solution(object):

    def minEatingSpeed(self, piles, H):
        if False:
            return 10
        '\n        :type piles: List[int]\n        :type H: int\n        :rtype: int\n        '

        def possible(piles, H, K):
            if False:
                return 10
            return sum(((pile - 1) // K + 1 for pile in piles)) <= H
        (left, right) = (1, max(piles))
        while left <= right:
            mid = left + (right - left) // 2
            if possible(piles, H, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left