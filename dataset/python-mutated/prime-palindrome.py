class Solution(object):

    def primePalindrome(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: int\n        '

        def is_prime(n):
            if False:
                i = 10
                return i + 15
            if n < 2 or n % 2 == 0:
                return n == 2
            return all((n % d for d in xrange(3, int(n ** 0.5) + 1, 2)))
        if 8 <= N <= 11:
            return 11
        for i in xrange(10 ** (len(str(N)) // 2), 10 ** 5):
            j = int(str(i) + str(i)[-2::-1])
            if j >= N and is_prime(j):
                return j