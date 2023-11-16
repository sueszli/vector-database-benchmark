class Solution(object):

    def superpalindromesInRange(self, L, R):
        if False:
            i = 10
            return i + 15
        '\n        :type L: str\n        :type R: str\n        :rtype: int\n        '

        def is_palindrome(k):
            if False:
                print('Hello World!')
            return str(k) == str(k)[::-1]
        K = int(10 ** ((len(R) + 1) * 0.25))
        (l, r) = (int(L), int(R))
        result = 0
        for k in xrange(K):
            s = str(k)
            t = s + s[-2::-1]
            v = int(t) ** 2
            if v > r:
                break
            if v >= l and is_palindrome(v):
                result += 1
        for k in xrange(K):
            s = str(k)
            t = s + s[::-1]
            v = int(t) ** 2
            if v > r:
                break
            if v >= l and is_palindrome(v):
                result += 1
        return result