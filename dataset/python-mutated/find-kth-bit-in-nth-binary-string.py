class Solution(object):

    def findKthBit(self, n, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        (flip, l) = (0, 2 ** n - 1)
        while k > 1:
            if k == l // 2 + 1:
                flip ^= 1
                break
            if k > l // 2:
                k = l + 1 - k
                flip ^= 1
            l //= 2
        return str(flip)