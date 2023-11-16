import collections

class Solution(object):

    def countTriplets(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        count_sum = collections.defaultdict(lambda : [0, 0])
        count_sum[0] = [1, 0]
        (result, prefix) = (0, 0)
        for (i, x) in enumerate(arr):
            prefix ^= x
            (c, t) = count_sum[prefix]
            result += c * i - t
            count_sum[prefix] = [c + 1, t + i + 1]
        return result