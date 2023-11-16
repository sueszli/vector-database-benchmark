class Solution(object):

    def findMaxAverage(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: float\n        '

        def getDelta(avg, nums, k):
            if False:
                return 10
            accu = [0.0] * (len(nums) + 1)
            minval_pos = None
            delta = 0.0
            for i in xrange(len(nums)):
                accu[i + 1] = nums[i] + accu[i] - avg
                if i >= k - 1:
                    if minval_pos == None or accu[i - k + 1] < accu[minval_pos]:
                        minval_pos = i - k + 1
                    if accu[i + 1] - accu[minval_pos] >= 0:
                        delta = max(delta, (accu[i + 1] - accu[minval_pos]) / (i + 1 - minval_pos))
            return delta
        (left, delta) = (min(nums), float('inf'))
        while delta > 1e-05:
            delta = getDelta(left, nums, k)
            left += delta
        return left