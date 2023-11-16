class Solution(object):

    def maxProduct(self, A):
        if False:
            i = 10
            return i + 15
        (global_max, local_max, local_min) = (float('-inf'), 1, 1)
        for x in A:
            (local_max, local_min) = (max(x, local_max * x, local_min * x), min(x, local_max * x, local_min * x))
            global_max = max(global_max, local_max)
        return global_max

class Solution2(object):

    def maxProduct(self, A):
        if False:
            return 10
        (global_max, local_max, local_min) = (float('-inf'), 1, 1)
        for x in A:
            local_max = max(1, local_max)
            if x > 0:
                (local_max, local_min) = (local_max * x, local_min * x)
            else:
                (local_max, local_min) = (local_min * x, local_max * x)
            global_max = max(global_max, local_max)
        return global_max