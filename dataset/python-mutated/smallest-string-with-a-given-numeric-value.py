class Solution(object):

    def getSmallestString(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        MAX_DIFF = ord('z') - ord('a')
        k -= n
        result = ['a'] * n
        for i in reversed(xrange(n)):
            tmp = min(k, MAX_DIFF)
            result[i] = chr(ord('a') + tmp)
            k -= tmp
            if k == 0:
                break
        return ''.join(result)