class Solution(object):

    def constructArray(self, n, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type k: int\n        :rtype: List[int]\n        '
        result = []
        (left, right) = (1, n)
        while left <= right:
            if k % 2:
                result.append(left)
                left += 1
            else:
                result.append(right)
                right -= 1
            if k > 1:
                k -= 1
        return result