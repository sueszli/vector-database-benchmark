class Solution(object):

    def kConcatenationMaxSum(self, arr, k):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :type k: int\n        :rtype: int\n        '

        def max_sub_k_array(arr, k):
            if False:
                while True:
                    i = 10
            (result, curr) = (float('-inf'), float('-inf'))
            for _ in xrange(k):
                for x in arr:
                    curr = max(curr + x, x)
                    result = max(result, curr)
            return result
        MOD = 10 ** 9 + 7
        if k == 1:
            return max(max_sub_k_array(arr, 1), 0) % MOD
        return (max(max_sub_k_array(arr, 2), 0) + (k - 2) * max(sum(arr), 0)) % MOD