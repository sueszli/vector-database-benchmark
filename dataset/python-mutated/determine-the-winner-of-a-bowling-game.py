class Solution(object):

    def isWinner(self, player1, player2):
        if False:
            i = 10
            return i + 15
        '\n        :type player1: List[int]\n        :type player2: List[int]\n        :rtype: int\n        '
        k = 2

        def f(arr):
            if False:
                return 10
            result = cnt = 0
            for i in xrange(len(arr)):
                result += 2 * arr[i] if cnt else arr[i]
                cnt += arr[i] == 10
                if i - k >= 0:
                    cnt -= arr[i - k] == 10
            return result
        (a, b) = (f(player1), f(player2))
        return 1 if a > b else 2 if a < b else 0