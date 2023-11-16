class Solution(object):

    def numSubarrayBoundedMax(self, A, L, R):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :type L: int\n        :type R: int\n        :rtype: int\n        '

        def count(A, bound):
            if False:
                return 10
            (result, curr) = (0, 0)
            for i in A:
                curr = curr + 1 if i <= bound else 0
                result += curr
            return result
        return count(A, R) - count(A, L - 1)