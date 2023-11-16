class Solution(object):

    def maximumCandies(self, candies, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type candies: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(x):
            if False:
                for i in range(10):
                    print('nop')
            return sum((c // x for c in candies)) >= k
        (left, right) = (1, max(candies))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right