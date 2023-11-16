class Solution(object):

    def shipWithinDays(self, weights, D):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type weights: List[int]\n        :type D: int\n        :rtype: int\n        '

        def possible(weights, D, mid):
            if False:
                print('Hello World!')
            (result, curr) = (1, 0)
            for w in weights:
                if curr + w > mid:
                    result += 1
                    curr = 0
                curr += w
            return result <= D
        (left, right) = (max(weights), sum(weights))
        while left <= right:
            mid = left + (right - left) // 2
            if possible(weights, D, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left