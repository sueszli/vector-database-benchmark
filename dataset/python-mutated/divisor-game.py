class Solution(object):

    def divisorGame(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: bool\n        '
        return N % 2 == 0

class Solution2(object):

    def divisorGame(self, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type N: int\n        :rtype: bool\n        '

        def memoization(N, dp):
            if False:
                print('Hello World!')
            if N == 1:
                return False
            if N not in dp:
                result = False
                for i in xrange(1, N + 1):
                    if i * i > N:
                        break
                    if N % i == 0:
                        if not memoization(N - i, dp):
                            result = True
                            break
                dp[N] = result
            return dp[N]
        return memoization(N, {})