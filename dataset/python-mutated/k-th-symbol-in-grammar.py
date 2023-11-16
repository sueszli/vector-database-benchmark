class Solution(object):

    def kthGrammar(self, N, K):
        if False:
            return 10
        '\n        :type N: int\n        :type K: int\n        :rtype: int\n        '

        def bitCount(n):
            if False:
                print('Hello World!')
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        return bitCount(K - 1) % 2