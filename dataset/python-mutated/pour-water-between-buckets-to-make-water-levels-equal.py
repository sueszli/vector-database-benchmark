class Solution(object):

    def equalizeWater(self, buckets, loss):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type buckets: List[int]\n        :type loss: int\n        :rtype: float\n        '

        def check(buckets, rate, x):
            if False:
                print('Hello World!')
            return sum((b - x for b in buckets if b - x > 0)) * rate >= sum((x - b for b in buckets if x - b > 0))
        EPS = 1e-05
        rate = (100 - loss) / 100.0
        (left, right) = (float(min(buckets)), float(sum(buckets)) / len(buckets))
        while right - left > EPS:
            mid = left + (right - left) / 2
            if not check(buckets, rate, mid):
                right = mid
            else:
                left = mid
        return left