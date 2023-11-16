class Solution(object):

    def maximumTripletValue(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        NEG_INF = float('-inf')
        result = 0
        mx_diff = mx = NEG_INF
        for x in nums:
            if mx_diff != NEG_INF:
                result = max(result, mx_diff * x)
            if mx != NEG_INF:
                mx_diff = max(mx_diff, mx - x)
            mx = max(mx, x)
        return result