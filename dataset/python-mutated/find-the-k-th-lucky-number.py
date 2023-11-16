class Solution(object):

    def kthLuckyNumber(self, k):
        if False:
            return 10
        '\n        :type k: int\n        :rtype: str\n        '
        result = []
        k += 1
        while k != 1:
            result.append('7' if k & 1 else '4')
            k >>= 1
        result.reverse()
        return ''.join(result)

class Solution2(object):

    def kthLuckyNumber(self, k):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :rtype: str\n        '
        return bin(k + 1)[3:].replace('1', '7').replace('0', '4')